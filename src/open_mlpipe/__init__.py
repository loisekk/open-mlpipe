"""open-mlpipe — Production-level automated ML pipeline.

Quick start:
    pip install open-mlpipe

    # CLI
    open-mlpipe run --data dataset.csv

    # Python API
    from open_mlpipe import run, PipelineConfig
    ctx = run("dataset.csv", target="price")

    # Config-driven
    from open_mlpipe import run_config
    ctx = run_config("configs/regression.yaml")
"""

__version__ = "1.0.3"

from open_mlpipe.config.schema import PipelineConfig
from open_mlpipe.core.pipeline import PipelineRunner


def run(data: str, target: str | None = None, project: str = "open-mlpipe"):
    """Run the full ML pipeline on a dataset.

    Args:
        data: Path to CSV/Parquet/Excel file
        target: Target column name (auto-detected if None)
        project: Project name for tracking

    Returns:
        PipelineContext with all results, metrics, and the trained model
    """
    from open_mlpipe.config.resolver import build_level1_config
    config = build_level1_config(data, target)
    config.project = project
    runner = PipelineRunner(config)
    return runner.run()


def run_config(config_path: str, project: str | None = None):
    """Run the full ML pipeline from a YAML config file.

    Args:
        config_path: Path to YAML config file
        project: Override project name (optional)

    Returns:
        PipelineContext with all results, metrics, and the trained model
    """
    from open_mlpipe.config.resolver import load_config, resolve_config
    from open_mlpipe.utils.io import load_data
    config = load_config(config_path)
    if project:
        config.project = project
    df = load_data(config.data.path)
    config = resolve_config(config, df)
    runner = PipelineRunner(config)
    return runner.run()


__all__ = [
    "__version__",
    "PipelineConfig",
    "PipelineRunner",
    "run",
    "run_config",
]
