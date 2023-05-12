import polars as pl
import pytest
from finta import TA

from finta_polars.indicators import OHLC_COLUMNS, simple_moving_average


@pytest.mark.benchmark(group="sma")
def test_simple_moving_average_price_all_prices_polars(ohlcv_df, benchmark):
    """Test that the simple moving average of the close price is correct."""
    sma = simple_moving_average(ohlcv_df, period=41)
    benchmark(sma.collect)


@pytest.mark.benchmark(group="sma_multiple_companies")
def test_simple_moving_average_price_all_prices_multiple_companies_polars(
    ohlcv_df_multiple_companies, benchmark
):
    """Test that the simple moving average of the close price is correct."""
    sma = simple_moving_average(
        ohlcv_df_multiple_companies, period=41, identifier_column="ticker"
    )
    benchmark(sma.collect)


@pytest.mark.benchmark(group="sma")
def test_simple_moving_average_price_close_finta(ohlcv_df, benchmark):
    """Test that the simple moving average of the close price is correct."""
    ohlc_df = ohlcv_df.to_pandas()
    benchmark(TA.SMA, ohlc_df, 41)


@pytest.mark.benchmark(group="sma")
def test_simple_moving_average_price_all_prices_naive_finta(ohlcv_df, benchmark):
    """Test that the simple moving average of the close price is correct."""
    ohlc_df = ohlcv_df.to_pandas()

    @benchmark
    def result():
        for c in OHLC_COLUMNS:
            TA.SMA(ohlc_df, 41, c)


@pytest.mark.benchmark(group="sma_multiple_companies")
def test_simple_moving_average_price_close_multiple_companies_finta(
    ohlcv_df_multiple_companies, benchmark
):
    """Test that the simple moving average of the close price is correct."""
    ohlc_df = ohlcv_df_multiple_companies.to_pandas()

    @benchmark
    def result():
        ohlc_df.groupby("ticker").apply(lambda df: TA.SMA(df, 41, "close"))
