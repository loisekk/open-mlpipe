# CLI Usage

## Commands

```bash
# Full pipeline (zero-touch)
open-mlpipe run --data dataset.csv

# With target specified
open-mlpipe run --data dataset.csv --target price

# With project name
open-mlpipe run --data dataset.csv --target price --project my-project

# Config-driven
open-mlpipe run --config configs/regression.yaml

# With deployment
open-mlpipe run --data dataset.csv --deploy

# EDA only (no training)
open-mlpipe profile --data dataset.csv

# Version
open-mlpipe --version

# Help
open-mlpipe --help
open-mlpipe run --help
```

## CLI Options

| Option | Description |
|--------|-------------|
| `--data, -d` | Path to data file (CSV, Parquet, Excel) |
| `--target, -t` | Target column name (auto-detected if not given) |
| `--level, -l` | Automation level (1=zero-touch, 2=config-driven) |
| `--config, -c` | Path to YAML config file |
| `--project, -p` | Project name |
| `--deploy` | Generate deployment artifacts |

## Examples

### Zero-Touch

```bash
open-mlpipe run --data dataset.csv
```

### With Target

```bash
open-mlpipe run --data dataset.csv --target price
```

### Config-Driven

```bash
open-mlpipe run --config configs/regression.yaml
```

### With Deployment

```bash
open-mlpipe run --data dataset.csv --deploy
```

## Next Steps

- [YAML Config](yaml-config.md)
- [Python API](python-api.md)
