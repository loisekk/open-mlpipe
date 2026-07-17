# Installation

## Requirements

- Python 3.10 or higher
- pip or conda

## Install from PyPI

```bash
pip install open-mlpipe
```

## Optional Extras

```bash
# CatBoost support
pip install open-mlpipe[catboost]

# MLflow tracking
pip install open-mlpipe[mlflow]

# Deployment (FastAPI + Docker)
pip install open-mlpipe[deploy]

# Everything
pip install open-mlpipe[full]
```

## Development Install

```bash
git clone https://github.com/loisekk/open-mlpipe.git
cd open-mlpipe
pip install -e ".[dev]"
```

## Verify Installation

```bash
python -c "import open_mlpipe; print(open_mlpipe.__version__)"
# 1.0.0
```

## Next Steps

- [Quick Start](quickstart.md)
- [Configuration](configuration.md)
