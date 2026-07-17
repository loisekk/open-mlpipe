"""Stage registry — maps stage names to implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from open_mlpipe.core.stage import Stage


class StageRegistry:
    """Registry for pipeline stages."""

    def __init__(self) -> None:
        self._stages: dict[str, Stage] = {}

    def register(self, stage: Stage) -> None:
        self._stages[stage.name] = stage

    def get(self, name: str) -> Stage:
        if name not in self._stages:
            available = ", ".join(sorted(self._stages.keys()))
            raise KeyError(f"Stage '{name}' not found. Available: {available}")
        return self._stages[name]

    def has(self, name: str) -> bool:
        return name in self._stages

    def list_stages(self) -> list[str]:
        return sorted(self._stages.keys())

    def default_order(self) -> list[str]:
        """Return stages in canonical execution order."""
        return [
            "load",
            "eda",
            "clean",
            "feature_eng",
            "split",
            "preprocess",
            "compare",
            "tune",
            "select",
            "evaluate",
            "explain",
            "save",
            "deploy",
        ]
