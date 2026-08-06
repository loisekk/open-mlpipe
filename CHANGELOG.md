# Changelog

All notable changes to **open-mlpipe** are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.8] — 2026-08-06

### Added
- **OpenCode-style splash screen** for `openml` interactive mode. Renders the
  ASCII-art logo, a bold horizontal rule with a version badge, a two-column
  key-bindings reference inside a bordered `Key Bindings` box (bright blue),
  and a `pip install open-mlpipe` install + tips footer inside a bordered
  `Get Started` box (bright green). All lines are scrollback-permanent (no
  alt-screen takeover) — the splash sits above every stage of the pipeline
  run and never disappears.

### Fixed
- **Pager rewritten with Rich `Live(screen=True)` alternate-screen app.**
  `openml show-log` / `view` / `show-log` now render in the terminal's
  alternate screen buffer (exactly what `less` / `vim` / Claude Code's
  inspectors do). On quit, the host terminal snaps back to the same
  scrollback position the user was at *before* opening the pager — the
  banner, every stage print, and the completion summary above are preserved
  untouched. No more cursor-juggling, no more escape-sequence leakage.
- **Post-run `View full output now?` prompt no longer auto-opens on Enter.**
  Default is now `N` (only `y`/`yes` opens the pager). Previously an empty
  Enter silently swallowed the user into the pager, which then captured the
  next `run` command's keystrokes and bled pipeline output into the alt
  screen. The prompt is also gated on TTY stdin so tests / piped stdin /
  CI never block.

### Removed
- No behavior regressions. All 113 unit tests passing.

## [1.0.7] — 2026-08-05

### Fixed
- Removed the Win32 `SetConsoleScreenBufferSize` hack
  (`_expand_buffer_now()` / `console_buffer`) that wiped scrollback on
  Windows Terminal, VS Code terminal, and conhost.
- Replaced `questionary.select().ask()` and `InquirerPy.list_prompt`
  full-screen pickers with a permanent-line numbered picker
  (`_pick_from_list`) to stop scrollback loss during dataset/target
  selection.
- Hardened `print_banner()` against non-TextIOWrapper stdout
  (piped subprocess, captured tests).
