"""DeployStage — generates FastAPI app + Dockerfile."""

from __future__ import annotations

from pathlib import Path

from open_mlpipe.core.context import PipelineContext
from open_mlpipe.core.stage import Stage


class DeployStage(Stage):
    name = "deploy"
    version = "1.0"

    def should_skip(self, ctx: PipelineContext) -> bool:
        return not ctx.config.deployment.enabled

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        output_dir = Path(ctx.config.artifacts.output_dir)

        # Generate FastAPI app
        self._generate_api(ctx, output_dir)

        # Generate Dockerfile
        self._generate_dockerfile(ctx, output_dir)

        return ctx

    def _generate_api(self, ctx, output_dir):
        """Generate FastAPI application."""
        # Build Pydantic model fields from X_train columns
        fields = []
        if ctx.X_train is not None:
            for col in ctx.X_train.columns:
                dtype = ctx.X_train[col].dtype
                if "int" in str(dtype):
                    fields.append(f"    {col}: int = 0")
                elif "float" in str(dtype):
                    fields.append(f"    {col}: float = 0.0")
                else:
                    fields.append(f'    {col}: str = ""')

        fields_str = "\n".join(fields)

        api_code = f'''"""Auto-generated FastAPI prediction endpoint."""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="{ctx.config.project} API")
pipe = joblib.load("{ctx.reports.get('model_path', 'artifacts/model_v1.joblib')}")


class InputFeatures(BaseModel):
{fields_str}


@app.post("/predict")
def predict(data: InputFeatures):
    df = pd.DataFrame([data.model_dump()])
    pred = pipe.predict(df)[0]
    result = {{"prediction": str(pred)}}

    try:
        prob = float(pipe.predict_proba(df).max())
        result["confidence"] = round(prob, 4)
    except Exception:
        pass

    return result


@app.get("/health")
def health():
    return {{"status": "ok", "model": "{ctx.best_model_name}"}}
'''
        api_path = output_dir / "main.py"
        api_path.write_text(api_code)

    def _generate_dockerfile(self, ctx, output_dir):
        """Generate Dockerfile."""
        docker_code = f'''FROM {ctx.config.deployment.docker_base_image}

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE {ctx.config.deployment.port}

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "{ctx.config.deployment.port}"]
'''
        docker_path = output_dir / "Dockerfile"
        docker_path.write_text(docker_code)
