"""Base stage interface — all pipeline stages implement this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from open_mlpipe.core.context import PipelineContext


class Stage(ABC):
    """Base class for all pipeline stages.

    Subclasses must define:
        name: ClassVar[str] = "stage_name"
        execute(ctx) -> PipelineContext
    """

    name: ClassVar[str] = "stage"
    version: ClassVar[str] = "1.0"

    @abstractmethod
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Run this stage. Read from ctx, modify ctx, return ctx."""

    def should_skip(self, ctx: PipelineContext) -> bool:
        """Override to conditionally skip this stage."""
        return False

    def __repr__(self) -> str:
        return f"<Stage:{self.name} v{self.version}>"
