"""Tests for utils/io.py — load_data and save_dataframe."""

from __future__ import annotations

import pandas as pd
import pytest

from open_mlpipe.utils.io import load_data, save_dataframe


@pytest.mark.unit
def test_load_data_csv(temp_csv: str):
    df = load_data(temp_csv)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "target" in df.columns


@pytest.mark.unit
def test_load_data_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_data("/nonexistent/file.csv")


@pytest.mark.unit
def test_save_dataframe_csv(sample_dataframe_regression: pd.DataFrame, temp_dir):
    import os

    path = os.path.join(str(temp_dir), "saved.csv")
    save_dataframe(sample_dataframe_regression, path)
    assert os.path.exists(path)
    reloaded = pd.read_csv(path)
    assert len(reloaded) == len(sample_dataframe_regression)
    assert list(reloaded.columns) == list(sample_dataframe_regression.columns)
