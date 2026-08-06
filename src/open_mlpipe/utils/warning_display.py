"""Warning display -- Rich Panel boxes with log file backup."""
from __future__ import annotations

import sys
import warnings
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console(width=80)

MAX_WARNINGS_IN_BOX = 3
LOG_DIR = Path("logs")


def fmt_metric(name: str, value) -> str:
    """Format a pipeline metric for display. Smart rounding by name + magnitude.

    - mape: shown as %, at most 2 dp
    - r2 / accuracy / f1 / roc_auc / mcc: 4 dp (small ranges)
    - rmse / mae / scores: round to 4 sig digits, group thousands on big values
    ponytail: callers used 4-dp for everything -> 8.9e15 mapele / 985579.5085 looked
    insane. One helper beat 6 per-site f-strings.
    """
    if not isinstance(value, (int, float)):
        return str(value)
    key = (name or "").lower()
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    # Percentage metric (lower=better). Cap insanity at scientific notation,
    # normal range is 0..100% so add the % suffix for production-style display.
    if "mape" in key:
        if abs(f) >= 1e6:
            return f"{f:.2e}"
        return f"{f:.2f}%"
    # Score-like 0..1 (or -inf..1 for r2): keep on 0..1 scale, 2 dp.
    # 2 dp surfaces overfit (1.00) and underfit (<0.50) at a glance.
    if any(k in key for k in ("r2", "accuracy", "f1", "roc_auc", "mcc")):
        return f"{f:.2f}"
    # Errors / counts / time: raw value in target units, 2 dp + thousands separators
    # (production print style: RMSE = 985,579.51 reads as $985K mismatch)
    if abs(f) >= 100:
        return f"{f:,.2f}"
    return f"{f:.4f}"


def fmt_compact(name: str, value) -> str:
    """Business-friendly abbreviation of large error/count metrics.

    Used in the model comparison table so execs can scan RMSE/MAE columns
    without counting digits: ``217,742.06`` -> ``217.7K``, ``1,180,000`` -> ``1.18M``.
    Score metrics (R2/Accuracy/F1/ROC_AUC/MCC) stay on the 0..1 scale -- they
    already read cleanly -- so this only touches error/loss/count metrics.

    Convention (Kaggle dashboards, Stripe / Datadog / Grafana defaults):
      < 1K      -> raw 2-dp                 217.74
      1K..1M    -> K suffix, 1 dp          217.7K
      1M..1B    -> M suffix, 2 dp          1.18M
      >= 1B     -> B suffix, 2 dp           2.40B
      NaN/inf   -> 'nan' / 'inf'           nan
    """
    if not isinstance(value, (int, float)):
        return str(value)
    key = (name or "").lower()
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    # Score-like 0..1 metrics stay 2-dp (R2 etc.) -- fmt_metric handles them,
    # but the comparison table routes everything through fmt_compact so keep
    # the same shape here for consistency.
    if any(k in key for k in ("r2", "accuracy", "f1", "roc_auc", "mcc")):
        return f"{f:.2f}"
    if "mape" in key:
        if abs(f) >= 1e6:
            return f"{f:.2e}"
        return f"{f:.2f}%"
    # NaN / inf
    if f != f:  # NaN
        return "nan"
    if f in (float("inf"), float("-inf")):
        return "inf"
    a = abs(f)
    if a >= 1e9:
        return f"{f/1e9:.2f}B"
    if a >= 1e6:
        return f"{f/1e6:.2f}M"
    if a >= 1e3:
        return f"{f/1e3:.1f}K"
    # Small enough to read raw -- 2 dp, no thousands group (no noise)
    if a >= 100:
        return f"{f:,.2f}"
    return f"{f:.2f}"

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


class stderr_capture:  # noqa: N801 - matches Win32 API capturing convention
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
            # Cosmetic warnings that are not actionable and just dirty the
            # pipeline output. Filtered here so they never reach the warning
            # panel. Add new patterns to this set as they surface.
            _cosmetic_markers = (
                "multivariate",            # Optuna TPE experimental feature
                "SettingWithCopy",         # pandas false-positive on .loc assigns
                "X does not have valid feature names",  # LightGBM sklearn API
                "artifact_path is deprecated",  # MLflow log_model rename
                "_readerthread",           # loky/joblib thread cleanup on Windows
                "Could not find the number of physical cores",  # joblib Win32
                "invalid value encountered in",  # numpy divide in MAPE (y_true~0)
                "divide by zero encountered in",  # numpy divide in MAPE
                "skops_trusted_types",      # MLflow skops security prompt
            )
            for w in w_list:
                msg = str(w.message)
                if any(marker in msg for marker in _cosmetic_markers):
                    continue
                cat = w.category.__name__ if hasattr(w.category, "__name__") else str(w.category)
                _collector.add(cat, msg, source=stage_name)


def display_warnings() -> None:
    _collector.display()


def clear_warnings() -> None:
    _collector.clear()


def save_warning_log() -> str | None:
    return _collector.save_log()


# ═══════════════════════════════════════════════════════════════════════════
# Windows console buffer management — prevents scrollback loss during pipeline
# ═══════════════════════════════════════════════════════════════════════════

# ── Windows console buffer management ───────────────────────────────────────
# REMOVED. Calling Win32 SetConsoleScreenBufferSize on the user's console
# destroys existing scrollback on Windows Terminal, VS Code terminal, and some
# conhost builds — this was the root cause of the "logo + previous output
# disappears at compare stage" bug.
#
# The correct pattern (used by Claude Code, Codex CLI, Gemini CLI, Aider, and
# every serious terminal app): never mutate the console. Just print permanent
# lines. Modern terminals allocate scrollback automatically.

_PIPE_LINES = 9999  # Kept for potential external imports; unused.


def _expand_buffer_now() -> None:
    """No-op retained for backward compatibility.

    Old callers may import this. It does nothing — the host terminal owns its
    own scrollback now.
    """
    return


class console_buffer:  # noqa: N801 - lowercase to mirror stderr_capture above
    """Deprecated no-op context manager.

    Previously expanded the Windows console scrollback buffer via Win32, which
    wiped existing scrollback on several terminal hosts. Now does nothing;
    it exists only so external code importing it doesn't break.
    New code should not use this.
    """

    def __enter__(self) -> console_buffer:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False
