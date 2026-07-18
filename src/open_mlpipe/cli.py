"""CLI entry point for open-mlpipe."""

from __future__ import annotations

import os
import sys
import time

# Fix Windows cp1252 UnicodeDecodeError in joblib/loky subprocesses
# Must be set before any joblib/sklearn imports
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

# Suppress loky/Windows UnicodeDecodeError in background threads
# This is a known Windows bug: wmic output is UTF-8 but subprocess reads cp1252
if sys.platform == "win32":
    import threading
    _orig_thread_run = threading.Thread.run

    def _patched_thread_run(self):
        try:
            _orig_thread_run(self)
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass  # loky cp1252 bug — non-fatal, falls back to logical cores

    threading.Thread.run = _patched_thread_run

    # Also patch subprocess to handle encoding errors
    import subprocess
    _orig_subprocess_run = subprocess.run

    def _patched_subprocess_run(*args, **kwargs):
        try:
            return _orig_subprocess_run(*args, **kwargs)
        except (UnicodeDecodeError, UnicodeEncodeError):
            # Return empty result on encoding error
            class FakeResult:
                returncode = 0
                stdout = b""
                stderr = b""
            return FakeResult()

    subprocess.run = _patched_subprocess_run

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns

console = Console()

# ═══════════════════════════════════════════════════════════════════════════
# ASCII ART BANNER
# ═══════════════════════════════════════════════════════════════════════════

BANNER = """
+=============================================================================+
|                                                                             |
|   ____   ___   ____  ____  ____  ____  ____  ____  ____  ____  ____  ____  |
|  |  _ \\ / _ \\ |  _ \\|  _ \\ | ___||  _ \\ | ___||  _ \\ | ___||  _ \\ | ___||  _ \\ |
|  | |_) | | | || | | || | | ||___ \\| | | ||___ \\| | | ||___ \\| | | ||___ \\| | | |
|  |  __/| |_| || |_| || |_| | ___) | |_| | ___) | |_| | ___) | |_| | ___) | |_| |
|  |_|    \\___/ |____/ |____/ |____/ \\___/ |____/ |____/ |____/ |____/ |____/ |____/ |
|                                                                             |
|              >_ open-mlpipe v1.0.2                                          |
|              Production ML Pipeline | 14+ Models | One Line                 |
|                                                                             |
+=============================================================================+
"""

BANNER_SIMPLE = """
[bold blue]>>> open-mlpipe v1.0.2[/bold blue]
[dim]Production ML Pipeline — 14+ Models — One Line[/dim]
"""


def print_banner():
    """Print the ASCII art banner."""
    # Force UTF-8 output for Unicode characters
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')
    print(BANNER)
    print()


def print_completion_summary(ctx, start_time):
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
            if k.startswith("test_") and isinstance(v, (int, float)):
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

    # Quick prediction example
    console.print()
    console.print("[dim]Quick prediction:[/dim]")
    console.print(
        "[dim]  import joblib[/dim]\n"
        "[dim]  model = joblib.load('{model_path}')[/dim]\n"
        "[dim]  predictions = model.predict(new_data)[/dim]".format(
            model_path=model_path
        )
    )


@click.group(invoke_without_command=True)
@click.version_option(package_name="open-mlpipe")
@click.pass_context
def main(ctx):
    """open-mlpipe — Production-level automated ML pipeline."""
    if ctx.invoked_subcommand is None:
        interactive_mode()


def interactive_mode():
    """Interactive mode - like Qwen Code."""
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

  [green]run[/green]      - Run the full ML pipeline
  [green]profile[/green]  - Profile a dataset (EDA only)
  [green]help[/green]     - Show this help
  [green]quit[/green]     - Exit

[bold]Quick Start:[/bold]
  > run --data dataset.csv --target column_name
  > profile --data dataset.csv
""")
                continue
            
            if user_input.lower() == "run":
                console.print("\n[bold]Run Pipeline[/bold]")
                console.print("[dim]Enter your data file path:[/dim]")
                data = console.input("[bold green]> data: [/bold green]").strip()
                if not data:
                    console.print("[red]No data file provided[/red]")
                    continue
                
                # Scan directory for datasets
                _scan_and_show_datasets(data)
                
                console.print("[dim]Enter target column (or press Enter for auto-detect):[/dim]")
                target = console.input("[bold green]> target: [/bold green]").strip() or None
                
                console.print(f"\n[bold]Starting pipeline for {data}...[/bold]\n")
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
            
            console.print(f"[red]Unknown command: {user_input}[/red]")
            console.print("[dim]Type 'help' for available commands[/dim]")
            
        except KeyboardInterrupt:
            console.print("\n[bold cyan]Goodbye! Happy ML! :)[/bold cyan]\n")
            break
        except EOFError:
            break


def _scan_and_show_datasets(path):
    """Scan directory for datasets and show columns."""
    from pathlib import Path
    import pandas as pd
    
    p = Path(path)
    
    # If it's a file, show its columns
    if p.is_file():
        try:
            df = pd.read_csv(p, nrows=5)
            console.print(f"\n[bold cyan]Dataset: {p.name}[/bold cyan]")
            console.print(f"  Rows: {len(pd.read_csv(p)):,}")
            console.print(f"  Columns ({len(df.columns)}):")
            for col in df.columns:
                dtype = df[col].dtype
                sample = df[col].iloc[0] if len(df) > 0 else "N/A"
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
                    rows = len(pd.read_csv(f, low_memory=False))
                    cols = len(df.columns)
                    
                    # Guess target candidates (numeric columns)
                    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
                    target_hint = ", ".join(num_cols[:3]) if num_cols else "None"
                    
                    table.add_row(str(i), f.name, str(rows), str(cols), target_hint)
                except Exception:
                    table.add_row(str(i), f.name, "?", "?", "?")
            
            console.print(table)
            console.print()
        else:
            console.print(f"\n[yellow]No CSV/Excel files found in {p}[/yellow]")
    else:
        console.print(f"\n[yellow]Path not found: {path}[/yellow]")


def _run_pipeline(data, target=None):
    """Run the ML pipeline."""
    from open_mlpipe.config.resolver import build_level1_config
    from open_mlpipe.core.pipeline import PipelineRunner
    
    start_time = time.time()
    pipeline_config = build_level1_config(data, target)
    pipeline_config.project = "open-mlpipe"
    
    runner = PipelineRunner(pipeline_config)
    ctx = runner.run()
    
    print_completion_summary(ctx, start_time)


def _profile_data(data, target=None):
    """Profile a dataset."""
    from open_mlpipe.config.schema import PipelineConfig
    from open_mlpipe.core.context import PipelineContext
    from open_mlpipe.stages.eda import EDALoaderStage
    from open_mlpipe.stages.load import DataLoaderStage
    
    config = PipelineConfig(
        project="profile",
        data={"path": data, "target": target},
    )
    
    ctx = PipelineContext(config=config)
    ctx = DataLoaderStage().execute(ctx)
    ctx = EDALoaderStage().execute(ctx)
    
    if ctx.eda_report:
        console.print("\n[bold cyan]=== EDA Report ===[/bold cyan]\n")
        
        quality = ctx.eda_report.get("quality", {})
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
        
        dist = ctx.eda_report.get("distributions", {})
        if dist.get("highly_skewed"):
            console.print(f"\n[bold]Highly Skewed:[/bold] {dist['highly_skewed']}")
        
        if ctx.statistical_tests is not None:
            console.print("\n[bold]Statistical Tests (significant features):[/bold]")
            sig = ctx.statistical_tests[ctx.statistical_tests["significant"]]
            for _, row in sig.iterrows():
                console.print(f"  {row['feature']}: {row['test']} (p={row['p_value']:.4f})")
        
        console.print("\n[bold green]Profiling complete![/bold green]")


if __name__ == "__main__":
    main()
