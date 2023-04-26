"""Tests for schemas.py."""
import polars as pl
import pytest

from finta_polars.schemas import validate_indicator_schema


def test_validate_indicator_schema():
    """Validate that a dataframe with the expected schema does not raise an error."""
    df = pl.DataFrame(
        {
            "open": [1.0, 2.0, 3.0],
            "high": [2.0, 3.0, 4.0],
            "low": [0.0, 1.0, 2.0],
            "close": [1.0, 2.0, 3.0],
        }
    ).lazy()
    validate_indicator_schema(df)


def test_validate_indicator_schema_with_volume():
    """Validate that a dataframe with the expected schema does not raise an error.

    This test runs with volume included.
    """
    df = pl.DataFrame(
        {
            "open": [1.0, 2.0, 3.0],
            "high": [2.0, 3.0, 4.0],
            "low": [0.0, 1.0, 2.0],
            "close": [1.0, 2.0, 3.0],
            "volume": [100, 200, 300],
        }
    ).lazy()
    validate_indicator_schema(df, include_volume=True)


def test_validate_indicator_schema_raises_error():
    """Validate that a dataframe with the wrong schema raises an error."""
    df = pl.DataFrame(
        {
            "high": [2.0, 3.0, 4.0],
            "low": [0.0, 1.0, 2.0],
            "close": [1.0, 2.0, 3.0],
            "volume": [100, 200, 300],
        }
    ).lazy()
    with pytest.raises(Exception):
        validate_indicator_schema(df)


def test_validate_indicator_schema_missing_volume_raises_error():
    """Validate that a dataframe with the wrong schema raises an error.

    This test runs with a missing volume.
    """
    df = pl.DataFrame(
        {
            "open": [1.0, 2.0, 3.0],
            "high": [2.0, 3.0, 4.0],
            "low": [0.0, 1.0, 2.0],
            "close": [1.0, 2.0, 3.0],
        }
    ).lazy()
    with pytest.raises(Exception):
        validate_indicator_schema(df, include_volume=True)
