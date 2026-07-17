"""CLI entry point for mlpipe."""

from __future__ import annotations

import os
import sys

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
        except UnicodeDecodeError:
            pass  # loky cp1252 bug — non-fatal, falls back to logical cores

    threading.Thread.run = _patched_thread_run

import click
from rich.console import Console

console = Console()


@click.group()
@click.version_option(package_name="mlpipe")
def main():
    """open-mlpipe — Production-level automated ML pipeline."""
    pass


@main.command()
@click.option("--data", "-d", default=None, help="Path to data file (CSV, Parquet, Excel)")
@click.option("--target", "-t", default=None, help="Target column name (auto-detected if not given)")
@click.option("--level", "-l", default=1, type=click.Choice(["1", "2", "3"]), help="Automation level")
@click.option("--config", "-c", default=None, help="Path to YAML config file (for level 2)")
@click.option("--project", "-p", default="mlpipe-run", help="Project name")
@click.option("--deploy/--no-deploy", default=False, help="Generate deployment artifacts")
def run(data, target, level, config, project, deploy):
    """Run the full ML pipeline."""
    from open_mlpipe.config.resolver import build_level1_config, load_config, resolve_config
    from open_mlpipe.core.pipeline import PipelineRunner
    from open_mlpipe.utils.io import load_data

    level = int(level)

    if config:
        console.print(f"[bold]Loading config from {config}...[/bold]")
        pipeline_config = load_config(config)
        # Load data to resolve auto values
        df = load_data(pipeline_config.data.path)
        pipeline_config = resolve_config(pipeline_config, df)
    elif data:
        if level == 1:
            console.print(f"[bold]Level 1: Zero-touch pipeline for {data}[/bold]")
            pipeline_config = build_level1_config(data, target)
        else:
            console.print("[red]Level 2 requires --config[/red]")
            return
    else:
        console.print("[red]Either --data or --config is required[/red]")
        return

    pipeline_config.project = project
    if not config:
        pipeline_config.level = level
    if deploy:
        pipeline_config.deployment.enabled = True

    runner = PipelineRunner(pipeline_config)
    ctx = runner.run()

    console.print("\n[bold green]Pipeline complete![/bold green]")
    console.print(f"Model saved to: {ctx.reports.get('model_path', 'N/A')}")


@main.command()
@click.option("--data", "-d", required=True, help="Path to data file")
@click.option("--target", "-t", default=None, help="Target column name")
def profile(data, target):
    """Profile a dataset (EDA only, no model training)."""
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

    # Print EDA summary
    if ctx.eda_report:
        console.print("\n[bold cyan]═══ EDA Report ═══[/bold cyan]\n")

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

        if ctx.vif_scores is not None:
            console.print("\n[bold]VIF (Variance Inflation Factor):[/bold]")
            for _, row in ctx.vif_scores.head(5).iterrows():
                flag = " ⚠️" if row["VIF"] > 10 else ""
                console.print(f"  {row['feature']}: {row['VIF']:.2f}{flag}")


if __name__ == "__main__":
    main()
