# Contributing

We welcome contributions to open-mlpipe! Here's how to get started.

## Development Setup

### Prerequisites

- Python 3.10+
- Git

### Clone and Install

```bash
git clone https://github.com/loisekk/open-mlpipe.git
cd open-mlpipe
pip install -e ".[dev]"
```

### Verify Setup

```bash
python -c "import open_mlpipe; print(open_mlpipe.__version__)"
```

## Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_pipeline.py

# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration
```

## Code Quality

open-mlpipe uses **ruff** for linting and formatting:

```bash
# Lint
ruff check .

# Format
ruff format .

# Fix auto-fixable issues
ruff check --fix .
```

## Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

## Project Structure

```
open-mlpipe/
├── src/open_mlpipe/          # Source code
│   ├── __init__.py           # Public API
│   ├── cli.py                # CLI entry point
│   ├── config/               # Configuration schemas
│   ├── core/                 # Pipeline runner, context, stages
│   ├── stages/               # Individual pipeline stages
│   └── utils/                # Utility functions
├── tests/                    # Test suite
├── configs/                  # Example YAML configs
├── docs/                     # Documentation source
└── examples/                 # Example scripts
```

## Adding a New Model

1. Create a new file in `src/open_mlpipe/stages/` or register in the model registry
2. Add the model configuration to the schema
3. Add tests for the new model
4. Update the documentation

## Adding a New Stage

1. Create a new stage class inheriting from `Stage`
2. Register it in the stage registry
3. Add configuration options to the schema
4. Add tests
5. Update documentation

## Documentation

Documentation is built with [MkDocs Material](https://squidfund.github.io/mkdocs-material/):

```bash
# Install docs dependencies
pip install mkdocs mkdocs-material mkdocstrings[python]

# Serve locally
mkdocs serve

# Build
mkdocs build
```

### Writing Style

- Use clear, concise language
- Include code examples for every feature
- Keep API references auto-generated from docstrings
- Test all code examples before committing

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests: `pytest`
5. Run linter: `ruff check .`
6. Commit with a clear message
7. Push and open a pull request

### PR Guidelines

- One feature or fix per PR
- Include tests for new functionality
- Update documentation if needed
- Follow existing code style
- Keep commits focused and atomic

## Reporting Issues

When reporting a bug, include:

1. Python version (`python --version`)
2. open-mlpipe version (`python -c "import open_mlpipe; print(open_mlpipe.__version__)"`)
3. Full error traceback
4. Minimal code to reproduce the issue
5. Operating system

## Code of Conduct

Be respectful, inclusive, and constructive. We're all here to build great tools together.

---

[Back to Home :octicons-arrow-right-24:](index.md)
