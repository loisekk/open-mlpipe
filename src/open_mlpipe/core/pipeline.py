"""Pipeline runner — orchestrates stage execution with tracking."""

from __future__ import annotations

import logging
import time

from rich.console import Console
from rich.table import Table

from open_mlpipe.config.resolver import resolve_config
from open_mlpipe.config.schema import PipelineConfig
from open_mlpipe.core.context import PipelineContext, StageMetadata
from open_mlpipe.core.registry import StageRegistry
from open_mlpipe.core.stage import Stage
from open_mlpipe.utils.warning_display import (
    capture_warnings,
    clear_warnings,
    display_warnings,
    save_warning_log,
    stderr_capture,
)

logger = logging.getLogger("mlpipe")
console = Console()


class PipelineRunner:
    """Executes a pipeline from config through all stages."""

    def __init__(self, config: PipelineConfig, registry: StageRegistry | None = None) -> None:
        self.config = config
        self.registry = registry or StageRegistry()
        self.ctx = PipelineContext(config=config)

    def run(self) -> PipelineContext:
        """Execute the full pipeline."""
        from open_mlpipe.stages import (
            CleanStage,
            CompareStage,
            DataLoaderStage,
            DeployStage,
            EDALoaderStage,
            EvaluateStage,
            ExplainStage,
            FeatureEngStage,
            PreprocessStage,
            SaveStage,
            SelectStage,
            SplitStage,
            TuneStage,
        )

        # Register all stages
        stages: list[Stage] = [
            DataLoaderStage(),
            EDALoaderStage(),
            CleanStage(),
            FeatureEngStage(),
            SplitStage(),
            PreprocessStage(),
            CompareStage(),
            TuneStage(),
            SelectStage(),
            EvaluateStage(),
            ExplainStage(),
            SaveStage(),
        ]

        if self.config.deployment.enabled:
            stages.append(DeployStage())

        for s in stages:
            self.registry.register(s)

        # Print header
        self._print_header()

        # Execute stages
        for stage in self.registry.default_order():
            if not self.registry.has(stage):
                continue
            s = self.registry.get(stage)
            if s.should_skip(self.ctx):
                console.print(f"  [dim]skip {stage}[/dim]")
                continue

            self._run_stage(s)

            # Resolve config after data is loaded (need raw_data for auto-detection)
            if stage == "load" and self.ctx.raw_data is not None:
                self.config = resolve_config(self.config, self.ctx.raw_data)
                self.ctx.config = self.config

        # Print summary
        self._print_summary()

        return self.ctx

    def _run_stage(self, stage: Stage) -> None:
        """Execute a single stage with timing and error handling."""
        console.print(f"\n[bold blue]>> {stage.name}[/bold blue]")

        start = time.time()
        try:
            capture_stderr = stage.name == "compare"
            with capture_warnings(stage_name=stage.name):
                if capture_stderr:
                    with stderr_capture(source=stage.name):
                        self.ctx = stage.execute(self.ctx)
                else:
                    self.ctx = stage.execute(self.ctx)
            elapsed = time.time() - start
            meta = StageMetadata(
                stage_name=stage.name,
                stage_version=stage.version,
                duration_seconds=elapsed,
            )
            self.ctx.add_stage(meta)
            console.print(f"  [green]OK {stage.name}[/green] [dim]({elapsed:.1f}s)[/dim]")
        except Exception as e:
            elapsed = time.time() - start
            console.print(f"  [red]FAIL {stage.name} failed after {elapsed:.1f}s: {e}[/red]")
            # Only re-raise for non-critical stages. Compare failure is
            # survivable — pipeline can continue with default model selection.
            if stage.name == "compare":
                logger.warning("Compare stage failed: %s. Pipeline continues.", e)
            else:
                raise
        finally:
            # Display warnings after each stage
            display_warnings()
            clear_warnings()

    def _print_header(self) -> None:
        console.print("\n[bold cyan]=== mlpipe Pipeline Runner ===[/bold cyan]")
        console.print(f"  Project:  [bold]{self.config.project}[/bold]")
        console.print(f"  Level:    {self.config.level}")
        console.print(f"  Data:     {self.config.data.path}")
        if self.config.data.target:
            console.print(f"  Target:   {self.config.data.target}")

    def _print_summary(self) -> None:
        from rich.panel import Panel
        from rich.text import Text

        # ── Model comparison table (R2 / Training R2 / MAE / RMSE / Accuracy / F1 / ROC_AUC) ──
        comparison = self.ctx.metrics.get("model_comparison") or {}
        valid = {k: v for k, v in comparison.items() if "per_metric" in v}
        if valid:
            # Determine which metric columns to show (intersection of available across models)
            metric_names: list[str] = []
            for v in valid.values():
                for m in v["per_metric"]:
                    if m not in metric_names:
                        metric_names.append(m)

            if metric_names:
                table = Table(
                    title="Model Comparison (cross-validated)",
                    show_header=True,
                    header_style="bold cyan",
                    border_style="blue",
                )
                table.add_column("Model", style="bold")
                # For each metric show <name> and "Train <name>" columns
                for m in metric_names:
                    table.add_column(m, justify="right")
                    table.add_column(f"Train {m}", justify="right", style="dim")

                # Sort by val R2 (or first val metric) descending so best is on top
                def _rank(item):
                    pm = item[1]["per_metric"]
                    if "R2" in pm:
                        return pm["R2"].get("val", float("-inf"))
                    if "Accuracy" in pm:
                        return pm["Accuracy"].get("val", float("-inf"))
                    return float("-inf")

                ranked = sorted(valid.items(), key=_rank, reverse=True)
                from open_mlpipe.utils.warning_display import fmt_compact
                for model_name, info in ranked:
                    pm = info["per_metric"]
                    row = [model_name]
                    for m in metric_names:
                        v = pm.get(m, {})
                        row.append(fmt_compact(m, v.get("val", float("nan"))))
                        row.append(fmt_compact(m, v.get("train", float("nan"))))
                    # Highlight best row
                    if model_name == self.ctx.best_model_name:
                        row[0] = f"[bold green]{model_name}[/bold green]"
                    table.add_row(*row)

                console.print()
                console.print(table)

        # Build summary content
        lines = Text()
        lines.append("Task                 ", style="bold")
        lines.append(f"{self.ctx.task_type}\n", style="cyan")

        lines.append("Target               ", style="bold")
        lines.append(f"{self.ctx.target_column}\n", style="cyan")

        lines.append("Model                ", style="bold")
        lines.append(f"{self.ctx.best_model_name}\n", style="green")

        lines.append("Stages               ", style="bold")
        lines.append(f"{len(self.ctx.stage_history)}\n", style="cyan")

        if self.ctx.metrics:
            from open_mlpipe.utils.warning_display import fmt_metric
            for k, v in self.ctx.metrics.items():
                if isinstance(v, int | float | str) and not isinstance(v, dict):
                    val = fmt_metric(k, v) if isinstance(v, float) else str(v)
                    # Color-code metrics
                    if "accuracy" in k or "f1" in k or "r2" in k or "roc" in k or "mcc" in k:
                        style = "green"
                    elif "overfit" in k.lower():
                        style = "yellow"
                    elif "rmse" in k or "mae" in k or "mape" in k:
                        style = "red"
                    else:
                        style = "white"
                    lines.append(f"{k:<20}", style="bold")
                    lines.append(f"{val}\n", style=style)

        # Save warning log
        log_path = save_warning_log()
        if log_path:
            lines.append("\nFull log: ", style="dim")
            lines.append(f"{log_path}", style="cyan")

        panel = Panel(
            lines,
            title="[bold bright_cyan]=== Pipeline Complete ===[/bold bright_cyan]",
            border_style="bright_cyan",
            padding=(0, 1),
            width=78,
        )

        console.print()
        console.print(panel)
