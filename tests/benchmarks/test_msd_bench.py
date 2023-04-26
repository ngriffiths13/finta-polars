import pytest
from finta import TA

from finta_polars.indicators import OHLC_COLUMNS, moving_std


@pytest.mark.benchmark(group="msd")
def test_simple_moving_std_price_all_prices_polars(ohlcv_df, benchmark):
    """Test that the simple moving average of the close price is correct."""
    msd = moving_std(ohlcv_df, period=41)
    benchmark(msd.collect)


@pytest.mark.benchmark(group="msd_multiple_companies")
def test_simple_moving_std_price_all_prices_multiple_companies_polars(
    ohlcv_df_multiple_companies, benchmark
):
    """Test that the simple moving average of the close price is correct."""
    msd = moving_std(ohlcv_df_multiple_companies, period=41, identifier_column="ticker")
    benchmark(msd.collect)


@pytest.mark.benchmark(group="msd")
def test_simple_moving_std_price_close_finta(ohlcv_df, benchmark):
    """Test that the simple moving average of the close price is correct."""
    ohlc_df = ohlcv_df.to_pandas()
    benchmark(TA.MSD, ohlc_df, 41)


@pytest.mark.benchmark(group="msd")
def test_simple_moving_std_price_all_prices_naive_finta(ohlcv_df, benchmark):
    """Test that the simple moving average of the close price is correct."""
    ohlc_df = ohlcv_df.to_pandas()

    @benchmark
    def result():
        for c in OHLC_COLUMNS:
            TA.MSD(ohlc_df, 41, c)


@pytest.mark.benchmark(group="msd_multiple_companies")
def test_simple_moving_std_price_close_multiple_companies_finta(
    ohlcv_df_multiple_companies, benchmark
):
    """Test that the simple moving average of the close price is correct."""
    ohlc_df = ohlcv_df_multiple_companies.to_pandas()

    @benchmark
    def result():
        ohlc_df.groupby("ticker").apply(lambda df: TA.MSD(df, 41, "close"))
