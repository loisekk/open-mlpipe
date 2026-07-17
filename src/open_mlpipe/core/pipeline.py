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
            raise

    def _print_header(self) -> None:
        console.print("\n[bold cyan]=== mlpipe Pipeline Runner ===[/bold cyan]")
        console.print(f"  Project:  [bold]{self.config.project}[/bold]")
        console.print(f"  Level:    {self.config.level}")
        console.print(f"  Data:     {self.config.data.path}")
        if self.config.data.target:
            console.print(f"  Target:   {self.config.data.target}")

    def _print_summary(self) -> None:
        console.print("\n[bold cyan]=== Pipeline Complete ===[/bold cyan]")

        table = Table(show_header=False, box=None)
        table.add_column("Key", style="bold")
        table.add_column("Value")

        table.add_row("Task", str(self.ctx.task_type))
        table.add_row("Target", str(self.ctx.target_column))
        table.add_row("Model", str(self.ctx.best_model_name))
        table.add_row("Stages", str(len(self.ctx.stage_history)))

        if self.ctx.metrics:
            for k, v in self.ctx.metrics.items():
                if isinstance(v, int | float | str) and not isinstance(v, dict):
                    table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))

        console.print(table)
