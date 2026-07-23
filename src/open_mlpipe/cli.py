"""CLI entry point for open-mlpipe."""

from __future__ import annotations

import os
import re
import sys
import time

from open_mlpipe.config.schema import DataConfig

# Fix Windows cp1252 UnicodeDecodeError in joblib/loky subprocesses
# Must be set before any joblib/sklearn imports
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

import click
from InquirerPy.resolver import prompt as inquirer_prompt
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from open_mlpipe import __version__

# Force Rich to use UTF-8 on Windows instead of legacy cp1252 rendering
console = Console(force_terminal=True, legacy_windows=False)
# Also reconfigure stdio for UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass  # If stderr is not a real TextIOWrapper (redirected), skip

_ANSI_STRIP = re.compile(r'\033\[[0-9;]*[a-zA-Z]')


class _SessionTee:
    """Write to both stdout AND a log file simultaneously.

    ANSI codes are stripped for the file so it stays readable in Notepad.
    Safely nested — only the outermost Tee closes the file.
    """

    def __init__(self, stdout, log_path) -> None:
        self.stdout = stdout
        self.log_file = open(log_path, "w", encoding="utf-8", errors="replace")
        self._depth = 0

    def write(self, text: str) -> None:
        self.stdout.write(text)
        self.log_file.write(_ANSI_STRIP.sub("", text))

    def flush(self) -> None:
        self.stdout.flush()
        self.log_file.flush()

    def close(self) -> None:
        if self.log_file is None or self.log_file.closed:
            return
        self.log_file.flush()
        self.log_file.close()

    @property
    def path(self) -> str:
        return self.log_file.name

# ═══════════════════════════════════════════════════════════════════════════
# ASCII ART BANNER
# ═══════════════════════════════════════════════════════════════════════════

BANNER_TEMPLATE = """
[bold rgb(247,42,0)] ██████╗ ██████╗ ███████╗███╗   ██╗███╗   ███╗██╗[/bold rgb(247,42,0)]     ┌──────────────────────────────────────────────────────────┐
[bold rgb(247,42,0)]██╔═══██╗██╔══██╗██╔════╝████╗  ██║████╗ ████║██║[/bold rgb(247,42,0)]     │ >_ OpenML Code (v{version})                                  │
[bold rgb(247,42,0)]██║   ██║██████╔╝█████╗  ██╔██╗ ██║██╔████╔██║██║[/bold rgb(247,42,0)]     │                                                          │
[bold rgb(247,42,0)]██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║██║╚██╔╝██║██║[/bold rgb(247,42,0)]     │ API Key | openml-python (pip) (/model to change)         │
[bold rgb(247,42,0)]╚██████╔╝██║     ███████╗██║ ╚████║██║ ╚═╝ ██║███████╗[/bold rgb(247,42,0)]│ ~                                                        │
[bold rgb(247,42,0)] ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝╚═╝     ╚═╝╚══════╝[/bold rgb(247,42,0)]└──────────────────────────────────────────────────────────┘
"""


def print_banner():
    """Print the ASCII art banner with orange color."""
    import sys

    from open_mlpipe import __version__
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')  # type: ignore[attr-defined]
    banner = BANNER_TEMPLATE.format(version=__version__)
    console.print(banner)
    console.print(f"[bold orange1]>_ openml v{__version__}[/bold orange1]")
    console.print("[dim]Production ML Pipeline | 14+ Models | One Line[/dim]")


def print_completion_summary(ctx, start_time, session_log_path=None):
    """Print a beautiful completion summary with stats."""
    elapsed = time.time() - start_time

    # Create results table
    table = Table(
        title="Pipeline Complete!",
        show_header=True,
        header_style="bold cyan",
        border_style="blue",
    )
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="green")

    # Basic info
    table.add_row("Task", str(ctx.task_type).split(".")[-1])
    table.add_row("Target", str(ctx.target_column))
    table.add_row("Best Model", str(ctx.best_model_name))
    table.add_row("Time", f"{elapsed:.1f}s")

    # Metrics
    if ctx.metrics:
        for k, v in ctx.metrics.items():
            if k.startswith("test_") and isinstance(v, int | float):
                if isinstance(v, float):
                    table.add_row(k, f"{v:.4f}")
                else:
                    table.add_row(k, str(v))

    console.print()
    console.print(table)

    # Model saved
    model_path = ctx.reports.get("model_path", "N/A")
    if model_path and model_path != "N/A":
        console.print()
        console.print(
            Panel(
                f"[green]Model saved to:[/green] [bold]{model_path}[/bold]",
                border_style="green",
            )
        )

    # Session log path
    if session_log_path:
        console.print()
        console.print(
            f"[dim]Full session log (scrollable):[/dim] [cyan]{session_log_path}[/cyan]"
        )

    # Quick prediction example
    console.print()
    console.print("[dim]Quick prediction:[/dim]")
    console.print(
        "[dim]  import joblib[/dim]\n"
        f"[dim]  model = joblib.load('{model_path}')[/dim]\n"
        "[dim]  predictions = model.predict(new_data)[/dim]"
    )


@click.group(invoke_without_command=True)
@click.version_option(prog_name="open-mlpipe", version=__version__)
@click.pass_context
def main(ctx):
    """open-mlpipe — Production-level automated ML pipeline."""
    if ctx.invoked_subcommand is None:
        interactive_mode()


@main.command()
@click.option("--data", "-d", required=True, help="Path to data file or directory")
@click.option("--target", "-t", default=None, help="Target column name")
@click.option("--project", "-p", default="openml", help="Project name")
def run(data: str, target: str | None, project: str) -> None:
    """Run the full ML pipeline — compare, tune, evaluate, save."""
    _run_pipeline(data, target, project)


@main.command()
@click.option("--data", "-d", required=True, help="Path to data file")
@click.option("--target", "-t", default=None, help="Target column name")
def profile(data: str, target: str | None) -> None:
    """Profile a dataset — EDA report with quality, distributions, stats."""
    _profile_data(data, target)


@main.command()
@click.option("--n", "-n", default=5, help="Number of logs to show")
def _list_logs(n: int = 5) -> None:
    """Helper: list recent pipeline session logs."""
    from pathlib import Path

    from open_mlpipe.utils.warning_display import LOG_DIR

    log_dir = Path(LOG_DIR)
    if not log_dir.exists():
        console.print("[yellow]No session logs found.[/yellow]")
        return

    log_files = sorted(log_dir.glob("pipeline_run_*.log"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not log_files:
        console.print("[yellow]No session logs found.[/yellow]")
        return

    table = Table(title=f"Recent Session Logs (last {len(log_files[:n])})", border_style="dim")
    table.add_column("#", style="dim")
    table.add_column("Timestamp", style="cyan")
    table.add_column("Size", style="green")
    table.add_column("Path", style="bright_blue")

    for i, f in enumerate(log_files[:n], 1):
        ts = f.stem.replace("pipeline_run_", "").replace("_", " ").replace("-", ":")
        size = f.stat().st_size
        size_str = f"{size:,} bytes" if size < 1024 else f"{size/1024:.1f} KB"
        table.add_row(str(i), ts, size_str, str(f))

    console.print()
    console.print(table)
    latest = log_files[0]
    console.print("\n[dim]View latest log (interactive pager):[/dim] [cyan]openml view[/cyan]")
    console.print(f"[dim]Open in Notepad:[/dim] [cyan]notepad {latest}[/cyan]")


def _show_log(log_file: str | None = None) -> None:
    """Helper: display a session log using the built-in interactive pager."""
    from pathlib import Path

    from open_mlpipe.utils.pager import _read_last_session_path, view_log
    from open_mlpipe.utils.warning_display import LOG_DIR

    log_dir = Path(LOG_DIR)
    if not log_dir.exists():
        console.print("[yellow]No session logs directory found.[/yellow]")
        return

    log_files = sorted(log_dir.glob("pipeline_run_*.log"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not log_files:
        console.print("[yellow]No session logs found.[/yellow]")
        return

    if log_file:
        matches = [f for f in log_files if log_file in f.name]
        if not matches:
            console.print(f"[red]No log file matching '{log_file}'. Use `openml logs` to list them.[/red]")
            return
        target = matches[0]
    else:
        # Prefer last session marker, fall back to latest file
        last = _read_last_session_path()
        target = last if last and last.exists() else log_files[0]

    # Show brief header, then launch pager
    console.print(f"\n[bold cyan]Viewing: {target.name} ({target.stat().st_size:,} bytes)[/bold cyan]")
    console.print("[dim]↑↓ scroll  / search  q quit  PgUp/PgDn page  g/G top/bottom[/dim]\n")

    # Brief pause so user reads header before pager takes over
    import time as _time
    _time.sleep(0.6)

    view_log(target)


@main.command()
@click.option("--n", "-n", default=5, help="Number of logs to show")
def logs(n: int) -> None:
    """List recent pipeline session logs."""
    _list_logs(n)


@main.command()
@click.argument("log_file", required=False)
def show_log(log_file: str | None) -> None:
    """Display a session log in the interactive pager.

    Pass a partial filename (e.g., pipeline_run_20260401_123045)
    or omit to view the most recent log.

    Pager keys: arrow keys scroll, / search, q quit, g/G top/bottom.
    """
    _show_log(log_file)


@main.command()
def view() -> None:
    """View the most recent pipeline session log in the interactive pager.

    Shortcut for `openml show-log` — opens the last run's full output.
    Pager keys: arrow keys scroll, / search, q quit, g/G top/bottom.
    """
    _show_log(None)


def interactive_mode():
    """Interactive mode - like Qwen Code."""
    from open_mlpipe.utils.warning_display import _expand_buffer_now
    _expand_buffer_now()  # Buffer 9999 BEFORE any output, so banner bhi scrollback me bache

    print_banner()

    console.print("[dim]Tips: Type 'run' to start pipeline, 'profile' for EDA, 'help' for commands, 'quit' to exit[/dim]\n")

    while True:
        try:
            user_input = console.input("[bold green]> [/bold green]").strip()

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "q"):
                console.print("\n[bold cyan]Goodbye! Happy ML! :)[/bold cyan]\n")
                break

            if user_input.lower() == "help":
                console.print("""
[bold]Available Commands:[/bold]

  [green]run[/green]         - Run the full ML pipeline
  [green]profile[/green]     - Profile a dataset (EDA only)
  [green]view[/green]        - View the most recent session log (interactive pager)
  [green]logs[/green]        - List recent session logs
  [green]show-log[/green]    - View a specific session log (interactive pager)
  [green]help[/green]        - Show this help
  [green]quit[/green]        - Exit

[bold]Quick Start:[/bold]
  > run --data dataset.csv --target column_name
  > profile --data dataset.csv
  > view   (after running a pipeline — browse all output)
""")
                continue

            if user_input.lower() == "run":
                console.print("\n[bold]Run Pipeline[/bold]")
                console.print("[dim]Enter your data file path (or directory):[/dim]")
                data = console.input("[bold green]> data: [/bold green]").strip()
                if not data:
                    console.print("[red]No data file provided[/red]")
                    continue

                # Scan directory for datasets
                from pathlib import Path
                p = Path(data)

                if p.is_dir():
                    # Interactive dataset selection
                    csv_files = list(p.glob("*.csv")) + list(p.glob("*.xlsx")) + list(p.glob("*.xls"))
                    if csv_files:
                        import questionary
                        choices = [f.name for f in csv_files]
                        selected = questionary.select(
                            "Select a dataset:",
                            choices=choices
                        ).ask()
                        if selected and selected.strip():
                            data = str(p / selected)
                        else:
                            console.print("[red]No dataset selected[/red]")
                            continue
                    else:
                        console.print(f"[red]No CSV/Excel files found in {data}[/red]")
                        continue
                elif p.is_file():
                    # Single file - show columns
                    _scan_and_show_datasets(data)
                else:
                    console.print(f"[red]Path not found: {data}[/red]")
                    continue

                console.print("[dim]Select target column:[/dim]")

                # Show column list and let user pick with arrow keys
                import pandas as pd
                try:
                    df_preview = pd.read_csv(data, nrows=100, low_memory=False)
                    cols = list(df_preview.columns)

                    # Auto-detect likely target columns
                    target_hints = _suggest_target_columns(df_preview)

                    # Build choices for InquirerPy
                    choices = []
                    for col in cols:
                        hint = ""
                        if col in [t["col"] for t in target_hints]:
                            hint_data = next(t for t in target_hints if t["col"] == col)
                            hint = f" [dim]({hint_data['reason']})[/dim]"
                        choices.append({"name": col, "value": col})

                    # Add auto-detect option
                    choices.insert(0, {"name": "[auto-detect] Let pipeline choose", "value": "__auto__"})

                    # Show suggestions
                    if target_hints:
                        console.print()
                        console.print("[bold cyan]Suggested target columns:[/bold cyan]")
                        for t in target_hints[:3]:
                            console.print(f"  [green]{t['col']}[/green] [dim]- {t['reason']}[/dim]")
                        console.print()

                    # Arrow key selection
                    result = inquirer_prompt({
                        "type": "list",
                        "name": "target",
                        "message": "Select target column:",
                        "choices": choices,
                        "default": choices[0]["value"] if target_hints else choices[1]["value"],
                    })

                    target = result["target"]
                    if target == "__auto__":
                        target = None
                        console.print("[dim]Using auto-detect[/dim]")
                    else:
                        console.print(f"[green]Selected: {target}[/green]")

                except Exception as e:
                    # Fallback to text input if anything fails
                    console.print(f"[yellow]Could not read columns: {e}[/yellow]")
                    target = None

                console.print(f"\n[bold]Starting pipeline for {Path(data).name}...[/bold]\n")
                _run_pipeline(data, target)
                continue

            if user_input.lower() == "profile":
                console.print("\n[bold]Profile Dataset[/bold]")
                console.print("[dim]Enter your data file path:[/dim]")
                data = console.input("[bold green]> data: [/bold green]").strip()
                if not data:
                    console.print("[red]No data file provided[/red]")
                    continue

                console.print(f"\n[bold]Profiling {data}...[/bold]\n")
                _profile_data(data, None)
                continue

            # Try to parse as a command
            parts = user_input.split()
            if parts[0].lower() == "run" and len(parts) > 1:
                # Parse run command arguments
                data = None
                target = None
                for i, part in enumerate(parts[1:], 1):
                    if part in ("--data", "-d") and i + 1 < len(parts):
                        data = parts[i + 1]
                    elif part in ("--target", "-t") and i + 1 < len(parts):
                        target = parts[i + 1]

                if data:
                    console.print(f"\n[bold]Starting pipeline for {data}...[/bold]\n")
                    _run_pipeline(data, target)
                else:
                    console.print("[red]Please provide --data parameter[/red]")
                continue

            if parts[0].lower() == "profile" and len(parts) > 1:
                data = None
                for i, part in enumerate(parts[1:], 1):
                    if part in ("--data", "-d") and i + 1 < len(parts):
                        data = parts[i + 1]

                if data:
                    console.print(f"\n[bold]Profiling {data}...[/bold]\n")
                    _profile_data(data, None)
                else:
                    console.print("[red]Please provide --data parameter[/red]")
                continue

            if user_input.lower() in ("view", "v"):
                from open_mlpipe.utils.pager import _read_last_session_path, view_log

                target = _read_last_session_path()
                if target and target.exists():
                    view_log(target)
                else:
                    console.print("[yellow]No session log found. Run a pipeline first.[/yellow]")
                continue

            if user_input.lower() in ("logs", "log"):
                _list_logs(n=5)
                continue

            if user_input.lower().startswith("show-log"):
                parts = user_input.split()
                log_arg = parts[1] if len(parts) > 1 else None
                _show_log(log_arg)
                continue

            console.print(f"[red]Unknown command: {user_input}[/red]")
            console.print("[dim]Type 'help' for available commands[/dim]")

        except KeyboardInterrupt:
            console.print("\n[bold cyan]Goodbye! Happy ML! :)[/bold cyan]\n")
            break
        except EOFError:
            break


def _suggest_target_columns(df):
    """Suggest likely target columns based on heuristics."""
    suggestions = []

    for col in df.columns:
        dtype = str(df[col].dtype)
        nunique = df[col].nunique()
        total = len(df)

        # Skip ID-like columns
        if nunique == total:
            continue
        if col.lower() in ("id", "index", "row", "record"):
            continue

        # Classification target: categorical or low-cardinality numeric
        if dtype == "object" or dtype == "category":
            if 2 <= nunique <= 20:
                suggestions.append({
                    "col": col,
                    "reason": f"Classification target ({nunique} categories)",
                })
            continue

        # Regression target: numeric with continuous values
        if dtype in ("float64", "int64"):
            # Check if it looks like a label/target
            is_continuous = nunique > 20
            has_outliers = False

            if is_continuous:
                q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    has_outliers = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum() > 0

            # Common target names
            target_names = ("target", "label", "y", "class", "output", "outcome", "prediction", "price", "value", "score", "amount", "total", "profit", "revenue")
            name_match = any(name in col.lower() for name in target_names)

            if name_match:
                suggestions.append({
                    "col": col,
                    "reason": "Possible target (name matches)",
                })
            elif nunique <= 10:
                suggestions.append({
                    "col": col,
                    "reason": f"Classification target ({nunique} unique values)",
                })

    # Sort by priority: name match first, then cardinality
    suggestions.sort(key=lambda x: (
        0 if "name matches" in x["reason"] else 1,
        x["col"],
    ))

    return suggestions


def _scan_and_show_datasets(path):
    """Scan directory for datasets and show columns."""
    from pathlib import Path

    import pandas as pd

    p = Path(path)

    # If it's a file, show its columns
    if p.is_file():
        try:
            df_full = pd.read_csv(p, low_memory=False)
            console.print(f"\n[bold cyan]Dataset: {p.name}[/bold cyan]")
            console.print(f"  Rows: {len(df_full):,}")
            console.print(f"  Columns ({len(df_full.columns)}):")
            for col in df_full.columns:
                dtype = df_full[col].dtype
                sample = df_full[col].iloc[0] if len(df_full) > 0 else "N/A"
                console.print(f"    - [green]{col}[/green] ({dtype}) [dim]sample: {sample}[/dim]")
            console.print()
            return
        except Exception as e:
            console.print(f"[red]Error reading file: {e}[/red]")
            return
    elif p.is_dir():
        # Scan directory for CSV/Excel files
        csv_files = list(p.glob("*.csv")) + list(p.glob("*.xlsx")) + list(p.glob("*.xls"))

        if csv_files:
            console.print(f"\n[bold cyan]Found {len(csv_files)} dataset(s) in {p.name}/[/bold cyan]\n")

            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("#", style="dim")
            table.add_column("File", style="green")
            table.add_column("Rows", justify="right")
            table.add_column("Columns", justify="right")
            table.add_column("Target Candidates")

            for i, f in enumerate(csv_files[:10], 1):  # Show max 10
                try:
                    df = pd.read_csv(f, nrows=100, low_memory=False)
                    # Count rows without full parse (fast line count)
                    try:
                        with open(f, encoding="utf-8") as fh:
                            row_count = sum(1 for _ in fh) - 1  # subtract header
                    except Exception:
                        row_count = len(pd.read_csv(f, nrows=1000, low_memory=False))
                    cols = len(df.columns)

                    # Guess target candidates (numeric columns)
                    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
                    target_hint = ", ".join(num_cols[:3]) if num_cols else "None"

                    table.add_row(str(i), f.name, str(row_count), str(cols), target_hint)
                except Exception:
                    table.add_row(str(i), f.name, "?", "?", "?")

            console.print(table)
            console.print()
        else:
            console.print(f"\n[yellow]No CSV/Excel files found in {p}[/yellow]")
    else:
        console.print(f"\n[yellow]Path not found: {path}[/yellow]")


def _run_pipeline(data, target=None, project="openml"):
    """Run the ML pipeline."""
    from datetime import datetime
    from pathlib import Path

    from open_mlpipe.config.resolver import build_level1_config
    from open_mlpipe.core.pipeline import PipelineRunner
    from open_mlpipe.utils.warning_display import (
        LOG_DIR,
        WarningCollector,
        console_buffer,
    )

    p = Path(data)

    # If it's a directory, find the first CSV file
    if p.is_dir():
        csv_files = list(p.glob("*.csv")) + list(p.glob("*.xlsx")) + list(p.glob("*.xls"))
        if csv_files:
            data = str(csv_files[0])
            console.print(f"[dim]Using first dataset: {csv_files[0].name}[/dim]")
        else:
            console.print(f"[red]No CSV/Excel files found in {data}[/red]")
            return

    # ── Open session log (Tee captures ALL output) ─────────────────────
    LOG_DIR.mkdir(exist_ok=True)
    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_log_path = LOG_DIR / f"pipeline_run_{session_ts}.log"
    tee = _SessionTee(sys.stdout, session_log_path)

    console.print(f"\n[dim]Session log: {tee.path}[/dim]")
    sys.stdout = tee

    start_time = time.time()
    ctx = None

    try:
        pipeline_config = build_level1_config(data, target)
        pipeline_config.project = project

        runner = PipelineRunner(pipeline_config)

        # Expand console buffer to 9999 lines during pipeline run
        import signal as _signal

        def _handle_interrupt(sig, frame):
            raise KeyboardInterrupt()

        _signal.signal(_signal.SIGINT, _handle_interrupt)

        with console_buffer():
            ctx = runner.run()

        print_completion_summary(ctx, start_time, session_log_path)

    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted by user. Cleaning up...[/yellow]")
        # Fall through to finally for cleanup

    except KeyError as e:
        tee.close()
        sys.stdout = tee.stdout  # restore before next print
        collector = WarningCollector("Pipeline Error")
        collector.add(
            "KeyError",
            f"Target column '{e}' not found in dataset. Check column names and try again.",
            source="pipeline",
        )
        collector.display()
        console.print("\n[dim]Hint: Run 'profile' command first to see available columns.[/dim]\n")

    except Exception as e:
        tee.close()
        sys.stdout = tee.stdout
        collector = WarningCollector("Pipeline Error")
        collector.add(
            type(e).__name__,
            f"{e}",
            source="pipeline",
        )
        collector.display()

    finally:
        if tee is not None:
            tee.close()
            sys.stdout = tee.stdout
            console.print(f"\n[dim]Full session log saved: {tee.path}[/dim]")

            # Auto-offer interactive log viewer
            from open_mlpipe.utils.pager import _save_last_session_path, view_log

            _save_last_session_path(Path(tee.path))

            try:
                console.print(
                    Panel(
                        "[bold green]v[/bold green] = view full log in interactive pager\n"
                        "  [dim]↑↓ scroll  / search  q quit  PgUp/PgDn page[/dim]",
                        title="[bold]View pipeline output[/bold]",
                        border_style="bright_green",
                    )
                )
            except UnicodeEncodeError:
                print("\nType 'view' or 'openml show-log' to browse the full log.")

            # Offer immediate viewing
            try:
                answer = console.input(
                    "\n[bold green]View full output now? (y/n)[/bold green] "
                ).strip().lower()
                if answer in ("y", "yes", ""):
                    view_log(tee.path)
            except (KeyboardInterrupt, EOFError):
                pass


def _profile_data(data, target=None):
    """Profile a dataset."""
    from pathlib import Path

    from open_mlpipe.config.schema import PipelineConfig
    from open_mlpipe.core.context import PipelineContext
    from open_mlpipe.stages.eda import EDALoaderStage
    from open_mlpipe.stages.load import DataLoaderStage
    from open_mlpipe.utils.warning_display import WarningCollector

    p = Path(data)

    # If it's a directory, find the first CSV/Excel file
    if p.is_dir():
        csv_files = list(p.glob("*.csv")) + list(p.glob("*.xlsx")) + list(p.glob("*.xls"))
        if not csv_files:
            collector = WarningCollector("Profile Error")
            collector.add(
                "FileNotFoundError",
                f"No CSV/Excel files found in {data}",
                source="profile",
            )
            collector.display()
            return
        data = str(csv_files[0])
        console.print(f"[dim]Using first dataset: {csv_files[0].name}[/dim]")

    if not Path(data).exists():
        collector = WarningCollector("Profile Error")
        collector.add(
            "FileNotFoundError",
            f"File not found: {data}",
            source="profile",
        )
        collector.display()
        return

    try:
        config = PipelineConfig(
            project="profile",
            data=DataConfig(path=data, target=target),
        )

        ctx = PipelineContext(config=config)
        ctx = DataLoaderStage().execute(ctx)
        ctx = EDALoaderStage().execute(ctx)

        report = ctx.eda_report
        if report:
            console.print("\n[bold cyan]=== EDA Report ===[/bold cyan]\n")

            quality = report.get("quality", {})
            if quality.get("missing_count"):
                console.print("[bold]Missing Values:[/bold]")
                for col, count in quality["missing_count"].items():
                    pct = quality["missing_pct"].get(col, 0)
                    console.print(f"  {col}: {count} ({pct}%)")

            if quality.get("n_duplicates"):
                console.print(f"\n[bold]Duplicates:[/bold] {quality['n_duplicates']} ({quality['duplicate_pct']}%)")

            if quality.get("outliers"):
                console.print("\n[bold]Outliers:[/bold]")
                for col, info in quality["outliers"].items():
                    console.print(f"  {col}: {info['count']} ({info['pct']}%)")

            dist = report.get("distributions", {})
            if dist.get("highly_skewed"):
                console.print(f"\n[bold]Highly Skewed:[/bold] {dist['highly_skewed']}")

            if ctx.statistical_tests is not None:
                console.print("\n[bold]Statistical Tests (significant features):[/bold]")
                sig = ctx.statistical_tests[ctx.statistical_tests["significant"]]
                for _, row in sig.iterrows():
                    console.print(f"  {row['feature']}: {row['test']} (p={row['p_value']:.4f})")

            console.print("\n[bold green]Profiling complete![/bold green]")

    except Exception as e:
        collector = WarningCollector("Profile Error")
        collector.add(
            type(e).__name__,
            str(e),
            source="profile",
        )
        collector.display()


if __name__ == "__main__":
    main()
