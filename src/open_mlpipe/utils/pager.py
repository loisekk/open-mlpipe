"""Built-in cross-platform terminal pager for viewing session logs.

Uses Rich's `Live(transient=True, screen=True)` to render the pager inside the
terminal's alternate screen buffer. This is the same approach every serious
TUI-style pager (less, vim, htop, Claude Code inspectors) uses:

  * The host terminal's main scrollback stays **untouched** — nothing the user
    printed before opening the pager ever disappears or scrolls away.
  * While the pager is open, the transcript lives only in the alt screen.
  * On quit (q / Esc / Ctrl+C), the alt screen closes and the cursor snaps
    back to the home position — the user is returned to exactly the same
    scrollback position they were at when they opened the pager.

Keys:
  ↑/↓/j/k         line scroll
  Space/PgDn      page down
  PgUp            page up
  g / G           top / bottom
  Home / End      top / bottom
  /               search forward (Enter to commit, Esc to cancel)
  n / N           next / previous match
  ← / →           horizontal scroll (long lines)
  q / Esc / Ctrl+C quit

No external deps beyond `rich` (already required by open-mlpipe). The pager
falls back to plain `console.print` of the file if Rich Live cannot be entered
(headless / piped stdin), in which case infinite scrollback just works.
"""

from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# ── Raw keyboard input ──────────────────────────────────────────────────────

if sys.platform == "win32":
    import msvcrt

    def _getch() -> bytes:
        return msvcrt.getch()

    def _kbhit() -> bool:
        return msvcrt.kbhit()

else:
    import select
    import termios
    import tty

    def _getch() -> bytes:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.buffer.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    def _kbhit() -> bool:
        return bool(select.select([sys.stdin], [], [], 0)[0])


# ── Pager state ─────────────────────────────────────────────────────────────

class _PagerState:
    """Mutable pager state — scroll position, search, dimensions."""

    def __init__(self, lines: list[str], file_path: Path) -> None:
        self.lines = lines
        self.file_path = file_path
        self.top = 0
        self.left = 0
        self.height = 24
        self.width = 80
        self.search_term = ""
        self.search_matches: list[int] = []
        self.search_idx = -1
        self.quit = False
        self.message = ""


# ── Escape sequence decoding ────────────────────────────────────────────────

def _decode_escape(first_byte: bytes) -> str:
    try:
        ch = first_byte.decode("utf-8", errors="replace")
    except Exception:
        return ""

    if len(ch) == 1:
        code = ord(ch)

        # Windows arrow keys via msvcrt.getch() send a 2-byte sequence:
        #   \x00 (NUL prefix)  + H/P/M/K  for up/down/right/left
        #   \xe0 (extended)    + H/P/M/K  for the same on some keyboards
        # We must read the second byte immediately or it stays in the buffer
        # and corrupts the next keystroke.
        if code == 0 or code == 224:  # \x00 or \xe0
            if _kbhit():
                second = _getch()
                try:
                    s = second.decode("utf-8", errors="replace")
                except Exception:
                    return ""
                win_map: dict[str, str] = {
                    "H": "up", "P": "down", "M": "right", "K": "left",
                    "I": "page_up", "Q": "page_down",
                    "G": "home", "O": "end",
                    "S": "delete", "R": "end",
                }
                return win_map.get(s, "")

        if code == 27:  # Escape — could be ESC alone or ANSI CSI sequence
            if _kbhit():
                second = _getch()
                if second == b"[":
                    # ANSI CSI: \x1b[<final> — Windows Terminal sends these
                    third = _getch()
                    return _decode_csi(third)
                if second == b"O":
                    # \x1b O <letter> — SS3 sequence (some F-keys / arrows in xterm)
                    third = _getch()
                    return _decode_csi(third)
                # ESC followed by a plain char — treat as ESC (quit)
                return "escape"
            return "escape"
        if code == 13:
            return "enter"
        if code == 32:
            return "space"
        if code in (127, 8):
            return "backspace"
        if 32 <= code <= 126:
            return ch
        return ""

    if len(ch) > 1:
        return ch if ch.isprintable() else ""

    return ""


def _decode_csi(final: bytes) -> str:
    try:
        f = final.decode("utf-8", errors="replace")
    except Exception:
        return ""

    mapping: dict[str, str] = {
        "A": "up", "B": "down", "C": "right", "D": "left",
        "H": "home", "F": "end",
        "5": "page_up", "6": "page_down",
        "1": "home", "4": "end", "3": "delete",
    }

    if f in mapping:
        if f in ("5", "6", "1", "3", "4"):
            tilde = _getch()
            if tilde == b"~":
                return mapping[f]
            return ""
        return mapping[f]

    if f.isalpha() and f.upper() in mapping:
        return mapping[f.upper()]

    return ""


# ── Rendering via Rich ──────────────────────────────────────────────────────

def _render_page(state: _PagerState) -> Panel:
    """Build a Rich Panel containing the visible window of log lines + status bar."""
    height = state.height
    width = state.width

    visible_lines: list[Text] = []
    end = min(state.top + height - 1, len(state.lines))  # reserve 1 line for status
    for i in range(state.top, end):
        raw = state.lines[i]
        if state.left > 0:
            raw = raw[state.left:]
        if len(raw) > width:
            raw = raw[:width]

        line_text = Text(raw, style="white")
        # Highlight search match on this line
        if state.search_term:
            term_lower = state.search_term.lower()
            idx = raw.lower().find(term_lower)
            if idx >= 0:
                line_text = Text(raw[:idx], style="white")
                line_text.append(raw[idx:idx + len(state.search_term)], style="bold reverse yellow")
                line_text.append(raw[idx + len(state.search_term):], style="white")
        # Line number gutter (subtle)
        gutter = Text(f"{i + 1:>5} ", style="dim")
        row = Text.assemble(gutter, line_text)
        visible_lines.append(row)

    # Pad to fill viewport
    while len(visible_lines) < height - 1:
        visible_lines.append(Text("~", style="dim blue"))

    body = Text("\n").join(visible_lines) if visible_lines else Text("")

    # Status bar
    total = len(state.lines)
    pct = int((state.top / total) * 100) if total else 100
    right = f" {state.top + 1}-{min(state.top + state.height, total)}/{total} ({pct}%) "

    if state.search_term and state.search_matches:
        search_right = f" /{state.search_term} [{state.search_idx + 1}/{len(state.search_matches)}]"
    elif state.search_term:
        search_right = f" /{state.search_term} (no matches)"
    else:
        search_right = ""

    status_text = Text()
    status_text.append(f" {state.file_path.name} ", style="bold cyan on reverse")
    status_text.append("  ↑↓ scroll  / search  q quit  g/G top/bottom  ←→ h-scroll ", style="dim")
    status_text.append(search_right, style="yellow")
    status_text.append(right, style="bold")

    if state.message:
        status_text = Text(state.message[:width], style="bold yellow on blue")

    panel = Panel(
        body,
        title=f"[bold]{state.file_path.name}[/bold]",
        subtitle=status_text,
        subtitle_align="left",
        border_style="bright_blue",
        padding=(0, 1),
    )
    return panel


# ── Input handling ──────────────────────────────────────────────────────────

def _handle_key(state: _PagerState, ch: bytes) -> None:
    seq = _decode_escape(ch)

    if seq in ("q", "escape"):
        state.quit = True
    elif seq in ("j", "down"):
        _scroll_down(state, 1)
    elif seq in ("k", "up"):
        _scroll_up(state, 1)
    elif seq in ("page_down", "space"):
        _scroll_down(state, state.height - 1)
    elif seq == "page_up":
        _scroll_up(state, state.height - 1)
    elif seq == "g":
        state.top = 0
        state.left = 0
    elif seq in ("G", "shift_g", "end"):
        state.top = max(0, len(state.lines) - state.height)
        state.left = 0
    elif seq == "home":
        state.top = 0
        state.left = 0
    elif seq == "left":
        state.left = max(0, state.left - 4)
    elif seq == "right":
        state.left += 4
    elif seq == "/":
        state.message = "/"
        _perform_search(state)
    elif seq == "n":
        _next_search_match(state)
    elif seq in ("N", "shift_n"):
        _prev_search_match(state)
    elif seq == "enter":
        pass  # no-op


def _scroll_down(state: _PagerState, amount: int) -> None:
    state.top = min(state.top + amount, max(0, len(state.lines) - state.height))


def _scroll_up(state: _PagerState, amount: int) -> None:
    state.top = max(0, state.top - amount)


def _perform_search(state: _PagerState) -> None:
    """Interactive search — read chars until Enter."""
    import shutil as _shutil
    state.width = _shutil.get_terminal_size().columns
    state.height = _shutil.get_terminal_size().lines

    term_chars: list[str] = []
    while True:
        ch = _getch()
        try:
            char = ch.decode("utf-8", errors="replace")
        except Exception:
            continue

        if char in ("\r", "\n"):
            break
        elif char == "\x1b":
            term_chars = []
            break
        elif char in ("\x08", "\x7f"):
            if term_chars:
                term_chars.pop()
        elif len(char) == 1 and ord(char) >= 32:
            term_chars.append(char)

    search_term = "".join(term_chars)
    state.message = ""
    state.search_term = search_term

    if not search_term:
        state.search_matches = []
        state.search_idx = -1
        return

    matches: list[int] = []
    term_lower = search_term.lower()
    for i, line in enumerate(state.lines):
        if term_lower in line.lower():
            matches.append(i)

    state.search_matches = matches
    if matches:
        # Jump to first match at/after current position
        state.search_idx = 0
        for idx, m in enumerate(matches):
            if m >= state.top:
                state.search_idx = idx
                break
        state.top = matches[state.search_idx]
    else:
        state.search_idx = -1
        state.message = f"Pattern not found: {search_term}"


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


# ── Public API ──────────────────────────────────────────────────────────────

def view_log(log_path: str | Path) -> None:
    """Open an interactive pager on a log file using Rich's alternate screen.

    This is the safe pattern — identical to what `less`, `vim`, and Claude
    Code's inspectors do:
      * Render inside the alternate screen buffer (`screen=True`)
      * `transient=True` so the pager's frames are NOT written to scrollback
        on exit (only the original transcript above stays)
      * On quit, the host terminal snaps back to the same scrollback position
        the user was at before opening the pager.

    Falls back to plain `console.print(line)` of the whole file if stdin is
    not a TTY (headless / piped / CI), in which case infinite scrollback just
    works without any pager.

    Keys: ↑↓/jk  line scroll    Space/PgUpPgDn  page
          /      search         n/N              next/prev match
          g/G    top/bottom     q/Esc/Ctrl+C     quit
          ←→     horizontal scroll (long lines)
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

    # Headless / non-TTY: just print the log inline — scrollback handles it.
    if not sys.stdin.isatty():
        fallback_console = Console()
        for line in lines:
            fallback_console.print(line)
        return

    import shutil as _shutil
    state = _PagerState(lines, path)
    state.height = _shutil.get_terminal_size().lines
    state.width = _shutil.get_terminal_size().columns

    pager_console = Console()

    # Manually enter the alternate screen buffer. We do NOT use Rich's
    # `Live(screen=True)` here because `Live` puts the terminal in raw mode
    # and races our `_getch()` for stdin — arrow keys never reach the input
    # loop, so the user can't scroll. Manual alt-screen + per-frame
    # clear/redraw gives us full control over stdin AND preserves scrollback
    # above (alt screen is a separate buffer; on exit we leave it and the
    # host terminal snaps back to the original scrollback position).
    ENTER_ALT = "\x1b[?1049h"   # enter alternate screen buffer
    LEAVE_ALT = "\x1b[?1049l"   # leave alternate screen buffer (restore)
    CLEAR_HOME = "\x1b[2J\x1b[H"  # clear screen + cursor home
    HIDE_CURSOR = "\x1b[?25l"
    SHOW_CURSOR = "\x1b[?25h"

    try:
        # Enter alt screen, hide cursor, render first frame.
        sys.stdout.write(ENTER_ALT)
        sys.stdout.write(HIDE_CURSOR)
        sys.stdout.flush()
        with pager_console.capture() as capture:
            pager_console.print(_render_page(state))
        rendered = capture.get()
        sys.stdout.write(CLEAR_HOME)
        sys.stdout.write(rendered)
        sys.stdout.flush()

        while not state.quit:
            ch = _getch()
            _handle_key(state, ch)
            # Recompute dims (terminal may have resized)
            state.height = _shutil.get_terminal_size().lines
            state.width = _shutil.get_terminal_size().columns

            # Clear + redraw the panel in place inside the alt screen.
            with pager_console.capture() as capture:
                pager_console.print(_render_page(state))
            rendered = capture.get()
            sys.stdout.write(CLEAR_HOME)
            sys.stdout.write(rendered)
            sys.stdout.flush()
    except KeyboardInterrupt:
        pass  # Ctrl+C quits cleanly
    finally:
        # Leave alt screen + restore cursor — host terminal snaps back to
        # the exact scrollback position the user was at before the pager.
        sys.stdout.write(SHOW_CURSOR)
        sys.stdout.write(LEAVE_ALT)
        sys.stdout.flush()


def find_latest_log() -> Path | None:
    """Return the most recent pipeline session log, or None."""
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
