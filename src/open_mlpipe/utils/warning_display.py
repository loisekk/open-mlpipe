"""Warning display -- Rich Panel boxes with log file backup."""
from __future__ import annotations

import sys
import warnings
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console(width=80)

MAX_WARNINGS_IN_BOX = 3
LOG_DIR = Path("logs")

EXPLANATIONS: dict[str, dict[str, str]] = {
    "mixed_types": {"label": "Mixed column types", "fix": "low_memory=False reads full CSV."},
    "small_sample": {"label": "Too few samples", "fix": "EDA skips stats on rare classes."},
    "divide_by_zero": {"label": "Divide-by-zero", "fix": "Constant-target groups excluded from R2."},
    "ohe_unknown": {"label": "OHE unseen cats", "fix": "handle_unknown='ignore' -- all-zero encoding."},
    "catboost_skip": {"label": "CatBoost missing", "fix": "pip install catboost"},
    "lgbm_names": {"label": "LGBM names", "fix": "Cosmetic -- model works."},
    "loky_cores": {"label": "Parallel info", "fix": "LOKY_MAX_CPU_COUNT env var set."},
    "deprecated": {"label": "Pandas deprecation", "fix": "Pipeline uses explicit params."},
    "unicode_error": {"label": "Windows cp1252", "fix": "Non-fatal -- falls back to logical cores."},
    "stderr_capture": {"label": "Stderr output", "fix": "See log file for full traceback."},
}


def _match(message: str) -> str | None:
    msg = message.lower()
    if "mixed types" in msg:
        return "mixed_types"
    if "sample" in msg and "small" in msg:
        return "small_sample"
    if "divide by zero" in msg:
        return "divide_by_zero"
    if "unknown categories" in msg or "transform" in msg:
        return "ohe_unknown"
    if "catboost" in msg:
        return "catboost_skip"
    if "feature names" in msg:
        return "lgbm_names"
    if "physical cores" in msg:
        return "loky_cores"
    if "deprecated" in msg:
        return "deprecated"
    if "cp1252" in msg or "charmap" in msg or "unicode" in msg:
        return "unicode_error"
    if "traceback" in msg or "error" in msg:
        return "stderr_capture"
    return None


class stderr_capture:
    """Context manager that redirects stderr to a temp file to capture threaded output.

    Replaces sys.stderr with a real file handle (opened with open()), so ALL
    threads — including loky _readerthread threads — write to the same file
    object. No fd-level operations (no os.dup2), so Windows handle invalidation
    cannot occur. After the block, reads the file back and saves output.

    Usage:
        with stderr_capture(source="compare") as cap:
            sklearn_function()
    """

    def __init__(self, source: str = "") -> None:
        self.source = source
        self.stderr_text: str = ""
        self._old_stderr: Any = None
        self._tmp_path: Path | None = None
        self._tmp_file: Any = None

    def __enter__(self) -> stderr_capture:
        # Create a real temp file — not StringIO — so threads have a real fd
        LOG_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = self.source.replace(" ", "_") if self.source else "stderr"
        self._tmp_path = LOG_DIR / f"stderr_{tag}_{ts}.log"
        self._tmp_file = open(self._tmp_path, "w", encoding="utf-8", errors="replace")

        # Swap sys.stderr to the real file
        self._old_stderr = sys.stderr
        sys.stderr = self._tmp_file

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        # ALWAYS restore original stderr, even on exception.
        # Do this BEFORE reading the file so threads see the real stderr.
        sys.stderr = self._old_stderr

        if self._tmp_file is not None:
            self._tmp_file.flush()
            self._tmp_file.close()

        # Read captured content from the temp file
        if self._tmp_path is not None and self._tmp_path.exists():
            self.stderr_text = self._tmp_path.read_text(encoding="utf-8", errors="replace")
            self._tmp_path.unlink()  # remove temp file

        if exc_type is not None:
            # Exception in wrapped code — propagate it. Don't add warnings.
            # The raw traceback IS the useful output here.
            return False

        if self.stderr_text.strip():
            self._save_to_log()
            _collector.add(
                category="stderr_capture",
                message=self.stderr_text.strip()[:200],
                source=self.source,
            )
            _collector.add_stderr_capture(self.stderr_text)

        return False

    def _save_to_log(self) -> None:
        # Already saved during __enter__ to self._tmp_path.
        # This is a no-op — the file already exists.
        pass


class WarningCollector:
    def __init__(self, title: str = "Pipeline Warnings") -> None:
        self.warnings: list[dict[str, str]] = []
        self.title = title
        self._log_lines: list[str] = []
        self._stderr_captures: list[str] = []

    def add(self, category: str, message: str, source: str = "") -> None:
        self.warnings.append({"category": category, "message": message.strip(), "source": source})

    def add_stderr_capture(self, text: str) -> None:
        self._stderr_captures.append(text)

    def clear(self) -> None:
        self.warnings.clear()
        self._stderr_captures.clear()

    @property
    def count(self) -> int:
        return len(self.warnings)

    def _log(self, text: str) -> None:
        self._log_lines.append(text)

    def display(self) -> None:
        if not self.warnings:
            return

        seen: dict[str, int] = Counter()
        unique: list[dict[str, str]] = []
        for w in self.warnings:
            key = w["message"][:60]
            seen[key] += 1
            if seen[key] == 1:
                unique.append(w)

        capped = unique[:MAX_WARNINGS_IN_BOX]
        hidden = len(unique) - len(capped)

        table = Table(show_header=True, box=None, padding=(0, 1), expand=False)
        table.add_column("Type", width=8, no_wrap=True)
        table.add_column("Warning", overflow="fold", max_width=55)
        table.add_column("x", width=4, justify="right", no_wrap=True)

        for w in capped:
            count = seen.get(w["message"][:60], 1)
            msg = w["message"][:80]
            suffix = str(count) if count > 1 else ""

            cat = w["category"]
            if "Dtype" in cat or "mixed" in msg.lower():
                badge = "[bold white on blue] DATA [/]"
            elif "SmallSample" in cat or "sample" in msg.lower():
                badge = "[bold white on yellow] STAT [/]"
            elif "divide by zero" in msg.lower() or "RuntimeWarning" in cat:
                badge = "[bold white on red] MATH [/]"
            elif "unknown categories" in msg.lower() or "transform" in msg.lower():
                badge = "[bold white on magenta] OHE [/]"
            elif "catboost" in msg.lower():
                badge = "[bold white on cyan] SKIP[/]"
            elif "feature names" in msg.lower():
                badge = "[bold white on cyan] LGBM[/]"
            elif "physical cores" in msg.lower():
                badge = "[bold white on cyan] JOB[/]"
            elif "cp1252" in msg.lower() or "charmap" in msg.lower():
                badge = "[bold white on red] CP12[/]"
            elif "traceback" in msg.lower() or "stderr" in cat.lower():
                badge = "[bold white on red] ERR [/]"
            else:
                badge = "[bold white on cyan] WARN[/]"

            table.add_row(badge, msg, suffix)

        if hidden > 0:
            table.add_row("[dim]...[/dim]", f"[dim]{hidden} more hidden[/dim]", "")

        shown_keys: list[str] = []
        for w in unique:
            key = _match(w["message"])
            if key and key not in shown_keys:
                shown_keys.append(key)

        fixes = "; ".join(
            f"{EXPLANATIONS[k]['label']}: {EXPLANATIONS[k]['fix']}" for k in shown_keys[:3]
        )

        panel_warnings = Panel(
            table,
            title=f"[bold]{self.title}[/bold] ({self.count} total, {len(unique)} unique)",
            border_style="bright_cyan",
            padding=(0, 0),
            width=78,
        )

        summary_text = f"[dim]{fixes}[/dim]" if fixes else ""

        panel_summary = Panel(
            summary_text,
            title="[bold]Fixes[/bold]",
            border_style="bright_green",
            padding=(0, 0),
            width=78,
        )

        console.print()
        console.print(panel_warnings)
        console.print(panel_summary)

        self._log(f"\n--- {self.title} ({self.count} total, {len(unique)} unique) ---")
        for w in capped:
            count = seen.get(w["message"][:60], 1)
            msg = w["message"][:80]
            suffix = f" x{count}" if count > 1 else ""
            self._log(f"  {msg}{suffix}")
        if hidden:
            self._log(f"  ...{hidden} more hidden")
        if fixes:
            self._log(f"  [Fixes] {fixes}")

        for i, capture in enumerate(self._stderr_captures):
            self._log(f"\n--- Stderr capture {i + 1} ---")
            self._log(capture)

    def save_log(self) -> str | None:
        """Save all logged warnings to a file."""
        if not self._log_lines:
            return None

        LOG_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = LOG_DIR / f"pipeline_warnings_{ts}.log"
        log_path.write_text("\n".join(self._log_lines), encoding="utf-8")
        return str(log_path)


_collector = WarningCollector()


def get_collector() -> WarningCollector:
    return _collector


@contextmanager
def capture_warnings(stage_name: str = ""):
    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always")
        try:
            yield
        except Exception:
            raise
        finally:
            for w in w_list:
                cat = w.category.__name__ if hasattr(w.category, "__name__") else str(w.category)
                _collector.add(cat, str(w.message), source=stage_name)


def display_warnings() -> None:
    _collector.display()


def clear_warnings() -> None:
    _collector.clear()


def save_warning_log() -> str | None:
    return _collector.save_log()


# ═══════════════════════════════════════════════════════════════════════════
# Windows console buffer management — prevents scrollback loss during pipeline
# ═══════════════════════════════════════════════════════════════════════════

import ctypes
import ctypes.wintypes

_PIPE_LINES = 9999
_RESTORE_LINES = 9001  # slightly less than max to avoid pushback on restore

# Win32 console API types
_SHORT = ctypes.wintypes.SHORT
_WORD = ctypes.wintypes.WORD
_DWORD = ctypes.wintypes.DWORD
_HANDLE = ctypes.wintypes.HANDLE
_STD_OUTPUT_HANDLE = _DWORD(-11)
_STD_ERROR_HANDLE = _DWORD(-12)


class _COORD(ctypes.Structure):
    _fields_ = [("X", _SHORT), ("Y", _SHORT)]


class _SMALL_RECT(ctypes.Structure):
    _fields_ = [
        ("Left", _SHORT),
        ("Top", _SHORT),
        ("Right", _SHORT),
        ("Bottom", _SHORT),
    ]


class _CONSOLE_SCREEN_BUFFER_INFO(ctypes.Structure):
    _fields_ = [
        ("dwSize", _COORD),
        ("dwCursorPosition", _COORD),
        ("wAttributes", _WORD),
        ("srWindow", _SMALL_RECT),
        ("dwMaximumWindowSize", _COORD),
    ]

_kernel32 = ctypes.windll.kernel32
_GetStdHandle = _kernel32.GetStdHandle
_GetConsoleScreenBufferInfo = _kernel32.GetConsoleScreenBufferInfo
_SetConsoleScreenBufferSize = _kernel32.SetConsoleScreenBufferSize


def _get_console_buffer_info(handle: int) -> _CONSOLE_SCREEN_BUFFER_INFO | None:
    csbi = _CONSOLE_SCREEN_BUFFER_INFO()
    if not _GetConsoleScreenBufferInfo(_HANDLE(handle), ctypes.byref(csbi)):
        return None  # Invalid handle (non-console context) — silently skip
    return csbi


def _set_console_buffer_height(handle: int, height: int) -> None:
    csbi = _get_console_buffer_info(handle)
    if csbi is None:
        return  # No console attached
    new_size = _COORD(_SHORT(csbi.dwSize.X), _SHORT(height))
    _SetConsoleScreenBufferSize(_HANDLE(handle), new_size)


class console_buffer:
    """Context manager that expands Windows console scrollback during a block.

    Sets the stdout and stderr console buffer to 9999 lines on enter,
    restores to 9001 on exit. On non-Windows or when not attached to
    a console (e.g., piping to file), this is a no-op.

    Usage:
        with console_buffer():
            run_pipeline()  # output stays fully scrollable
    """

    def __init__(self) -> None:
        self._old_stdout_height: int | None = None
        self._old_stderr_height: int | None = None
        self._stdout_handle: int | None = None
        self._stderr_handle: int | None = None

    def __enter__(self) -> console_buffer:
        try:
            self._stdout_handle = _GetStdHandle(_STD_OUTPUT_HANDLE)
            if self._stdout_handle:
                csbi = _get_console_buffer_info(self._stdout_handle)
                if csbi is not None:
                    self._old_stdout_height = csbi.dwSize.Y
                    _set_console_buffer_height(self._stdout_handle, _PIPE_LINES)
        except OSError:
            self._stdout_handle = None

        try:
            self._stderr_handle = _GetStdHandle(_STD_ERROR_HANDLE)
            if self._stderr_handle and self._stderr_handle != self._stdout_handle:
                _set_console_buffer_height(self._stderr_handle, _PIPE_LINES)
        except OSError:
            self._stderr_handle = None

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self._stdout_handle and self._old_stdout_height is not None:
            try:
                _set_console_buffer_height(self._stdout_handle, _RESTORE_LINES)
            except OSError:
                pass
        return False
