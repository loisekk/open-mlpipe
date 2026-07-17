from __future__ import annotations

from open_mlpipe.stages.deploy import DeployStage


class TestDeployStage:
    def test_name(self):
        assert DeployStage.name == "deploy"

    def test_should_skip_when_disabled(self, tuned_context_regression):
        ctx = tuned_context_regression
        ctx.config.deployment.enabled = False
        assert DeployStage().should_skip(ctx) is True

    def test_execute_generates_main_py(self, tuned_context_regression, temp_dir):
        ctx = tuned_context_regression
        ctx.config.deployment.enabled = True
        ctx.config.artifacts.output_dir = str(temp_dir)
        ctx.reports["model_path"] = "model_v1.joblib"
        DeployStage().execute(ctx)
        assert (temp_dir / "main.py").exists()

    def test_execute_generates_dockerfile(self, tuned_context_regression, temp_dir):
        ctx = tuned_context_regression
        ctx.config.deployment.enabled = True
        ctx.config.artifacts.output_dir = str(temp_dir)
        ctx.reports["model_path"] = "model_v1.joblib"
        DeployStage().execute(ctx)
        assert (temp_dir / "Dockerfile").exists()
