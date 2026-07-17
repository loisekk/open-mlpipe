from __future__ import annotations

import json

from mlpipe.stages.save import SaveStage


class TestSaveStage:
    def test_name(self):
        assert SaveStage.name == "save"

    def test_execute_saves_model(self, tuned_context_regression, temp_dir):
        ctx = tuned_context_regression
        ctx.config.artifacts.output_dir = str(temp_dir)
        SaveStage().execute(ctx)
        assert (temp_dir / "model_v1.joblib").exists()

    def test_execute_creates_metadata(self, tuned_context_regression, temp_dir):
        ctx = tuned_context_regression
        ctx.config.artifacts.output_dir = str(temp_dir)
        SaveStage().execute(ctx)
        assert (temp_dir / "metadata.json").exists()

    def test_metadata_has_expected_keys(self, tuned_context_regression, temp_dir):
        ctx = tuned_context_regression
        ctx.config.artifacts.output_dir = str(temp_dir)
        SaveStage().execute(ctx)
        with open(temp_dir / "metadata.json") as f:
            meta = json.load(f)
        assert "project" in meta
        assert "task" in meta
        assert "target" in meta
        assert "best_model" in meta
