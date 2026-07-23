"""Built-in cross-platform terminal pager for viewing session logs.

No external dependencies — works on Windows (msvcrt), Linux/macOS (termios/tty).
Used by `openml show-log` and interactive `view` command to browse full logs
without losing scrollback to the host terminal emulator's buffer limits.

Key features:
- Arrow keys / PageUp / PageDown to scroll
- / to search forward, n/N for next/previous match
- g/G to jump to start/end
- q or Escape to quit
- Handles long lines (truncates to terminal width)
- Works in piped/subprocess contexts (where CONOUT$ fails)

Architecture:
- Reads the full log file into memory (lines list)
- Draws a fit-to-terminal-height window of lines
- Uses raw terminal input via msvcrt (Windows) or tty.setraw (Unix)
- ANSI codes for clearing and redrawing the visible portion in-place
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

# ── Raw keyboard input ──────────────────────────────────────────────────────

if sys.platform == "win32":
    import msvcrt

    def _getch() -> bytes:
        """Read one byte from stdin without echo or buffering."""
        return msvcrt.getch()

    def _kbhit() -> bool:
        """Check if a keypress is waiting."""
        return msvcrt.kbhit()

else:
    import termios
    import tty

    def _getch() -> bytes:
        """Read one byte from stdin in raw mode."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.buffer.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    def _kbhit() -> bool:
        """Non-blocking check — always returns False on Unix.
        Alternative: select.select([sys.stdin], [], [], 0).
        For simplicity, we block-read one char at a time on Unix.
        """
        import select
        return bool(select.select([sys.stdin], [], [], 0)[0])


# ── Terminal dimensions ─────────────────────────────────────────────────────

def _terminal_height() -> int:
    """Get terminal height, falling back to 24 if unreadable."""
    try:
        size = shutil.get_terminal_size()
        return max(size.lines, 5)
    except Exception:
        return 24


def _terminal_width() -> int:
    """Get terminal width, falling back to 80."""
    try:
        size = shutil.get_terminal_size()
        return max(size.columns, 40)
    except Exception:
        return 80


# ── ANSI helpers ────────────────────────────────────────────────────────────

_CLEAR_SCREEN = "\033[2J"
_CLEAR_LINE = "\033[2K"
_CURSOR_HOME = "\033[H"
_CURSOR_HIDE = "\033[?25l"
_CURSOR_SHOW = "\033[?25h"
_RESET = "\033[0m"
_REVERSE = "\033[7m"
_BOLD = "\033[1m"
_DIM = "\033[2m"


def _move_cursor(row: int, col: int) -> str:
    """ANSI escape to move cursor to (row, col) — 1-indexed."""
    return f"\033[{row};{col}H"


# ── Pager state ─────────────────────────────────────────────────────────────

class _PagerState:
    """Mutable state for the pager — scroll position, search, dimensions."""

    def __init__(self, lines: list[str], file_path: Path) -> None:
        self.lines = lines
        self.file_path = file_path
        self.top = 0  # First visible line index
        self.left = 0  # Horizontal scroll offset (for long lines)
        self.height = _terminal_height() - 2  # minus status bar
        self.width = _terminal_width()
        self.search_term = ""
        self.search_matches: list[int] = []
        self.search_idx = -1  # Current match index within search_matches
        self.quit = False
        self.message = ""  # Status bar message


# ── Drawing ─────────────────────────────────────────────────────────────────

def _draw_screen(state: _PagerState) -> None:
    """Redraw the visible portion of the pager screen."""
    height = state.height
    width = state.width

    # Build visible lines
    visible: list[str] = []
    for i in range(state.top, min(state.top + height, len(state.lines))):
        raw = state.lines[i]
        # Truncate to terminal width (minus left scroll)
        if state.left > 0:
            raw = raw[state.left:]
        if len(raw) > width:
            raw = raw[:width]
        visible.append(raw)

    # Pad if fewer lines than screen height
    while len(visible) < height:
        visible.append("~")

    # Build status bar
    status = _build_status_bar(state)

    # Build full screen output
    out_lines: list[str] = []
    out_lines.append(_CURSOR_HIDE)

    for row_idx, line in enumerate(visible):
        out_lines.append(_move_cursor(row_idx + 1, 1))
        out_lines.append(_CLEAR_LINE)

        # Highlight search matches
        if state.search_term and state.search_matches:
            actual_line_idx = state.top + row_idx
            if actual_line_idx in state.search_matches:
                # Highlight the matching portion (case-insensitive)
                highlighted = _highlight_search_match(line, state.search_term)
                out_lines.append(highlighted)
                out_lines.append(_move_cursor(row_idx + 1, 1))
                continue

        out_lines.append(line)

    # Status bar at bottom
    out_lines.append(_move_cursor(height + 1, 1))
    out_lines.append(_CLEAR_LINE)
    out_lines.append(_REVERSE)
    out_lines.append(status)
    out_lines.append(_RESET)

    out_lines.append(_move_cursor(height + 1, 1))

    sys.stdout.write("".join(out_lines))
    sys.stdout.flush()


def _build_status_bar(state: _PagerState) -> str:
    """Build the reverse-video status bar line."""
    width = state.width
    pct = 0
    if state.lines:
        pct = int((state.top / len(state.lines)) * 100) if state.lines else 100
    total = len(state.lines)

    left = f" {state.file_path.name}  "
    right = f" {state.top + 1}-{min(state.top + state.height, total)}/{total} ({pct}%) "

    if state.search_term:
        search_info = f" /{state.search_term}"
        if state.search_matches:
            search_info += f" [{state.search_idx + 1}/{len(state.search_matches)}]"
        center = search_info
    else:
        center = ""

    # Build bar: left + padding + center + padding + right
    mid_space = width - len(left) - len(right) - 2
    if len(center) > mid_space:
        center = center[:mid_space]
    pad_left = max(0, (mid_space - len(center)) // 2)
    pad_right = max(0, mid_space - len(center) - pad_left)

    if state.message:
        return state.message[:width]

    return f"{left}{' ' * pad_left}{center}{' ' * pad_right}{right}"


def _highlight_search_match(line: str, term: str) -> str:
    """Return line with first occurrence of term highlighted using ANSI reverse."""
    if not term:
        return line
    line_lower = line.lower()
    term_lower = term.lower()
    idx = line_lower.find(term_lower)
    if idx < 0:
        return line
    before = line[:idx]
    match = line[idx : idx + len(term)]
    after = line[idx + len(term) :]
    return f"{before}{_REVERSE}{_BOLD}{match}{_RESET}{after}"


# ── Input handling ──────────────────────────────────────────────────────────

def _handle_key(state: _PagerState, ch: bytes) -> None:
    """Process a single keypress or escape sequence."""
    # Parse escape sequences
    seq = _decode_escape(ch)

    if seq == "q" or seq == "escape":
        state.quit = True
    elif seq == "j" or seq == "down":
        _scroll_down(state, 1)
    elif seq == "k" or seq == "up":
        _scroll_up(state, 1)
    elif seq == "down":
        _scroll_down(state, 1)
    elif seq == "up":
        _scroll_up(state, 1)
    elif seq == "page_down" or seq == "space":
        _scroll_down(state, state.height)
    elif seq == "page_up":
        _scroll_up(state, state.height)
    elif seq == "g":
        if state.search_term == "g":  # double-g
            state.top = 0
            state.left = 0
            state.search_term = ""
        else:
            state.search_term = "g"  # first g, wait for second
    elif seq == "shift_g" or seq == "G":
        state.top = max(0, len(state.lines) - state.height)
        state.left = 0
    elif seq == "home":
        state.top = 0
        state.left = 0
    elif seq == "end":
        state.top = max(0, len(state.lines) - state.height)
    elif seq == "left":
        state.left = max(0, state.left - 4)
    elif seq == "right":
        state.left += 4  # capped by truncation in draw
    elif seq == "/":
        state.message = "/"
        _draw_screen(state)
        _perform_search(state)
    elif seq == "n":
        _next_search_match(state)
        state.search_term = ""  # clear one-char search
    elif seq == "shift_n" or seq == "N":
        _prev_search_match(state)
        state.search_term = ""  # clear one-char search
    else:
        # If in one-char search mode (waiting for second 'g')
        if state.search_term == "g" and seq:
            pass  # already handled above


def _scroll_down(state: _PagerState, amount: int) -> None:
    state.top = min(state.top + amount, max(0, len(state.lines) - state.height))


def _scroll_up(state: _PagerState, amount: int) -> None:
    state.top = max(0, state.top - amount)


def _perform_search(state: _PagerState) -> None:
    """Interactive search: read chars until Enter, then find matches."""
    term_chars: list[str] = []
    while True:
        ch = _getch()
        try:
            char = ch.decode("utf-8", errors="replace")
        except Exception:
            continue

        if char == "\r" or char == "\n":  # Enter
            break
        elif char == "\x1b":  # Escape — cancel search
            term_chars = []
            break
        elif char == "\x08" or char == "\x7f":  # Backspace
            if term_chars:
                term_chars.pop()
                state.message = f"/{''.join(term_chars)}"
                _draw_screen(state)
        elif len(char) == 1 and ord(char) >= 32:
            term_chars.append(char)
            state.message = f"/{''.join(term_chars)}"
            _draw_screen(state)

    search_term = "".join(term_chars)
    state.message = ""
    state.search_term = search_term

    if not search_term:
        state.search_matches = []
        state.search_idx = -1
        return

    # Find all matches (case-insensitive)
    matches: list[int] = []
    term_lower = search_term.lower()
    for i, line in enumerate(state.lines):
        if term_lower in line.lower():
            matches.append(i)

    state.search_matches = matches
    if matches:
        # Jump to first match after current position
        state.search_idx = 0
        for idx, m in enumerate(matches):
            if m >= state.top:
                state.search_idx = idx
                break
        state.top = matches[state.search_idx]
    else:
        state.search_idx = -1
        state.message = f"Pattern not found: {search_term}"
        _draw_screen(state)
        # Brief message, then clear
        _getch()  # wait for any key
        state.message = ""


def _next_search_match(state: _PagerState) -> None:
    if not state.search_matches:
        state.message = "No active search"
        return
    state.search_idx = (state.search_idx + 1) % len(state.search_matches)
    state.top = state.search_matches[state.search_idx]


def _prev_search_match(state: _PagerState) -> None:
    if not state.search_matches:
        return
    state.search_idx = (state.search_idx - 1) % len(state.search_matches)
    state.top = state.search_matches[state.search_idx]


# ── Escape sequence decoding ────────────────────────────────────────────────

def _decode_escape(first_byte: bytes) -> str:
    """Decode a keypress into a named action string."""
    try:
        ch = first_byte.decode("utf-8", errors="replace")
    except Exception:
        return ""

    # Single characters
    if len(ch) == 1:
        code = ord(ch)
        if code == 27:  # Escape
            # Check if there's more bytes waiting (arrow keys etc.)
            if _kbhit():
                second = _getch()
                if second == b"[":
                    third = _getch()
                    return _decode_csi(third)
                else:
                    # Alt+key or lone Escape
                    return "escape"
            return "escape"
        if code == 13:  # Enter
            return "enter"
        if code == 32:  # Space
            return "space"
        if code == 127 or code == 8:  # Backspace
            return "backspace"
        if 32 <= code <= 126:
            return ch
        return ""

    # Multi-byte UTF-8
    if len(ch) > 1:
        return ch if ch.isprintable() else ""

    return ""


def _decode_csi(final: bytes) -> str:
    """Decode CSI (Control Sequence Introducer) sequences: arrow keys, etc."""
    try:
        f = final.decode("utf-8", errors="replace")
    except Exception:
        return ""

    mapping: dict[str, str] = {
        "A": "up",
        "B": "down",
        "C": "right",
        "D": "left",
        "H": "home",
        "F": "end",
        "5": "page_up",   # CSI 5~
        "6": "page_down",  # CSI 6~
        "1": "home",       # CSI 1~  (some terminals)
        "4": "end",        # CSI 4~  (some terminals)
        "3": "delete",     # CSI 3~
    }

    if f in mapping:
        # For PageUp/Down, Home/End — check if there's a tilde
        if f in ("5", "6", "1", "3", "4"):
            tilde = _getch()
            if tilde == b"~":
                return mapping[f]
            return ""
        return mapping[f]

    # Handle shift-modified (CSI 1;2A etc.) — decode "A" at end as "up"
    if f.isalpha() and f.upper() in mapping:
        return mapping[f.upper()]

    return ""


# ── Public API ──────────────────────────────────────────────────────────────

def view_log(log_path: str | Path) -> None:
    """Open an interactive pager on a log file.

    Args:
        log_path: Path to the session log file to view.

    Use keys:
        ↑↓/jk — line scroll      Space/PgUp/PgDn — page scroll
        / — search                n/N — next/prev match
        g — top                   G — bottom
        q/Esc — quit
    """
    path = Path(log_path)
    if not path.exists():
        print(f"Log file not found: {path}", file=sys.stderr)
        return

    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"Failed to read log: {e}", file=sys.stderr)
        return

    lines = raw.splitlines()
    if not lines:
        print("(empty log)", file=sys.stderr)
        return

    state = _PagerState(lines, path)

    # Clear screen and draw initial view
    sys.stdout.write(_CLEAR_SCREEN)
    sys.stdout.write(_CURSOR_HOME)
    _draw_screen(state)

    try:
        while not state.quit:
            ch = _getch()
            _handle_key(state, ch)
            _draw_screen(state)
    except KeyboardInterrupt:
        pass  # Ctrl+C quits cleanly
    finally:
        # Restore terminal
        sys.stdout.write(_CURSOR_SHOW)
        sys.stdout.write(_RESET)
        sys.stdout.write(_move_cursor(state.height + 2, 1))
        sys.stdout.write("\n")
        sys.stdout.flush()


def find_latest_log() -> Path | None:
    """Return the most recent pipeline session log, or None.

    Searches the LOG_DIR for pipeline_run_*.log files, sorted by modification time.
    """
    from open_mlpipe.utils.warning_display import LOG_DIR

    log_dir = Path(LOG_DIR)
    if not log_dir.exists():
        return None

    log_files = sorted(
        log_dir.glob("pipeline_run_*.log"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    return log_files[0] if log_files else None


def _save_last_session_path(log_path: Path) -> None:
    """Store the last session log path so `view` always finds the latest."""
    from open_mlpipe.utils.warning_display import LOG_DIR

    LOG_DIR.mkdir(exist_ok=True)
    marker = LOG_DIR / ".last_session"
    try:
        marker.write_text(str(log_path), encoding="utf-8")
    except Exception:
        pass  # Best-effort — failure shouldn't block the pipeline


def _read_last_session_path() -> Path | None:
    """Read the stored last session path, falling back to find_latest_log."""
    from open_mlpipe.utils.warning_display import LOG_DIR

    marker = LOG_DIR / ".last_session"
    try:
        if marker.exists():
            stored = Path(marker.read_text(encoding="utf-8").strip())
            if stored.exists():
                return stored
    except Exception:
        pass
    return find_latest_log()
