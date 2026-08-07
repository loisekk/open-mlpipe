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

__version__ = "1.0.9"

try:
    from open_mlpipe.config.schema import PipelineConfig
    from open_mlpipe.core.pipeline import PipelineRunner
except ImportError as _e:
    # pandas/numpy DLL load failures (Python 3.14 on locked-down Windows boxes,
    # AV Application Control policies) surface here as ImportError. Give a human
    # message + fix instead of a raw traceback.
    msg = getattr(_e, "msg", str(_e)) or ""
    if "DLL load failed" in msg or "pandas" in msg or "numpy" in msg:
        import sys as _sys
        _sys.stderr.write(
            "\n[open-mlpipe] Failed to load a core dependency (pandas/numpy).\n"
            f"  Reason: {msg}\n\n"
            "  This is usually an environment issue, not a bug in open-mlpipe:\n"
            "    1. Antivirus / Application Control policy blocking the DLL\n"
            "       -> exclude the Python install dir, or run from a clean venv\n"
            "    2. Python 3.14 + pandas/numpy wheel mismatch\n"
            "       -> `pip install --upgrade pandas numpy` (or use Python 3.12/3.13)\n"
            "    3. Mixed Python installs on PATH\n"
            "       -> use a dedicated venv: `python -m venv .venv && .\\.venv\\Scripts\\activate`\n\n"
        )
        raise SystemExit(2) from _e
    raise


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
