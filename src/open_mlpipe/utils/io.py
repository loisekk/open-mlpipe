from pathlib import Path

import pandas as pd


def load_data(path: str, **kwargs) -> pd.DataFrame:
    """Auto-detect format and load data."""
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    suffix = p.suffix.lower()

    loaders = {
        ".csv": lambda: pd.read_csv(p, low_memory=False, **kwargs),
        ".parquet": lambda: pd.read_parquet(p, **kwargs),
        ".xlsx": lambda: pd.read_excel(p, **kwargs),
        ".xls": lambda: pd.read_excel(p, **kwargs),
        ".json": lambda: pd.read_json(p, **kwargs),
    }

    if suffix not in loaders:
        raise ValueError(f"Unsupported file format: {suffix}. Supported: {list(loaders.keys())}")

    return loaders[suffix]()


def save_dataframe(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to file based on suffix."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    suffix = p.suffix.lower()
    if suffix == ".csv":
        df.to_csv(p, index=False)
    elif suffix == ".parquet":
        df.to_parquet(p, index=False)
    elif suffix in (".xlsx", ".xls"):
        df.to_excel(p, index=False)
    else:
        raise ValueError(f"Unsupported save format: {suffix}")
