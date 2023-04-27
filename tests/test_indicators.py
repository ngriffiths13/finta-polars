"""Tests for indicator functions."""
import polars as pl
import pytest

from finta_polars.indicators import (
    moving_std,
    simple_moving_average,
    simple_moving_median,
)


def test_simple_moving_average_no_volume(ohlcv_df):
    ohlc_df = ohlcv_df.drop("volume")
    out = simple_moving_average(ohlc_df, period=5).collect()
    assert out.shape == (3000, 4)
    assert out.select(pl.last("close_sma_5")).item() == 2997.0
    assert out.columns == ["open_sma_5", "high_sma_5", "low_sma_5", "close_sma_5"]


def test_simple_moving_average_volume(ohlcv_df):
    out = simple_moving_average(ohlcv_df, period=5).collect()
    assert out.shape == (3000, 5)
    assert out.select(pl.last("volume_sma_5")).item() == 2997.0
    assert out.columns == [
        "open_sma_5",
        "high_sma_5",
        "low_sma_5",
        "close_sma_5",
        "volume_sma_5",
    ]


def test_simple_moving_average_multiple_companies(ohlcv_df_multiple_companies):
    out = simple_moving_average(ohlcv_df_multiple_companies, period=5).collect()
    assert out.shape == (15000, 6)
    assert out.select(pl.last("volume_sma_5")).item() == 2997.0
    assert out.columns == [
        "ticker",
        "open_sma_5",
        "high_sma_5",
        "low_sma_5",
        "close_sma_5",
        "volume_sma_5",
    ]


def test_simple_moving_median_no_volume(ohlcv_df):
    ohlc_df = ohlcv_df.drop("volume")
    out = simple_moving_median(ohlc_df, period=5).collect()
    assert out.shape == (3000, 4)
    assert out.select(pl.last("close_smm_5")).item() == 2997.0
    assert out.columns == ["open_smm_5", "high_smm_5", "low_smm_5", "close_smm_5"]


def test_simple_moving_median_volume(ohlcv_df):
    out = simple_moving_median(ohlcv_df, period=5).collect()
    assert out.shape == (3000, 5)
    assert out.select(pl.last("volume_smm_5")).item() == 2997.0
    assert out.columns == [
        "open_smm_5",
        "high_smm_5",
        "low_smm_5",
        "close_smm_5",
        "volume_smm_5",
    ]


def test_simple_moving_median_multiple_companies(ohlcv_df_multiple_companies):
    out = simple_moving_median(ohlcv_df_multiple_companies, period=5).collect()
    assert out.shape == (15000, 6)
    assert out.select(pl.last("volume_smm_5")).item() == 2997.0
    assert out.columns == [
        "ticker",
        "open_smm_5",
        "high_smm_5",
        "low_smm_5",
        "close_smm_5",
        "volume_smm_5",
    ]


def test_simple_moving_std_no_volume(ohlcv_df):
    ohlc_df = ohlcv_df.drop("volume")
    out = moving_std(ohlc_df, period=5).collect()
    assert out.shape == (3000, 4)
    assert pytest.approx(out.select(pl.last("close_msd_5")).item(), 0.0001) == 1.58113
    assert out.columns == ["open_msd_5", "high_msd_5", "low_msd_5", "close_msd_5"]


def test_simple_moving_std_volume(ohlcv_df):
    out = moving_std(ohlcv_df, period=5).collect()
    assert out.shape == (3000, 5)
    assert pytest.approx(out.select(pl.last("volume_msd_5")).item(), 0.0001) == 1.58113
    assert out.columns == [
        "open_msd_5",
        "high_msd_5",
        "low_msd_5",
        "close_msd_5",
        "volume_msd_5",
    ]


def test_simple_moving_std_multiple_companies(ohlcv_df_multiple_companies):
    out = moving_std(ohlcv_df_multiple_companies, period=5).collect()
    assert out.shape == (15000, 6)
    assert pytest.approx(out.select(pl.last("volume_msd_5")).item(), 0.0001) == 1.58113
    assert out.columns == [
        "ticker",
        "open_msd_5",
        "high_msd_5",
        "low_msd_5",
        "close_msd_5",
        "volume_msd_5",
    ]
