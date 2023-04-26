import pytest
from finta import TA

from finta_polars.indicators import OHLC_COLUMNS, simple_moving_median


@pytest.mark.benchmark(group="smm")
def test_simple_moving_median_price_all_prices_polars(ohlcv_df, benchmark):
    """Test that the simple moving average of the close price is correct."""
    smm = simple_moving_median(ohlcv_df, period=41)
    benchmark(smm.collect)


@pytest.mark.benchmark(group="smm_multiple_companies")
def test_simple_moving_median_price_all_prices_multiple_companies_polars(
    ohlcv_df_multiple_companies, benchmark
):
    """Test that the simple moving average of the close price is correct."""
    smm = simple_moving_median(
        ohlcv_df_multiple_companies, period=41, identifier_column="ticker"
    )
    benchmark(smm.collect)


@pytest.mark.benchmark(group="smm")
def test_simple_moving_median_price_close_finta(ohlcv_df, benchmark):
    """Test that the simple moving average of the close price is correct."""
    ohlc_df = ohlcv_df.to_pandas()
    benchmark(TA.SMM, ohlc_df, 41)


@pytest.mark.benchmark(group="smm")
def test_simple_moving_median_price_all_prices_naive_finta(ohlcv_df, benchmark):
    """Test that the simple moving average of the close price is correct."""
    ohlc_df = ohlcv_df.to_pandas()

    @benchmark
    def result():
        for c in OHLC_COLUMNS:
            TA.SMM(ohlc_df, 41, c)


@pytest.mark.benchmark(group="smm_multiple_companies")
def test_simple_moving_median_price_close_multiple_companies_finta(
    ohlcv_df_multiple_companies, benchmark
):
    """Test that the simple moving average of the close price is correct."""
    ohlc_df = ohlcv_df_multiple_companies.to_pandas()

    @benchmark
    def result():
        ohlc_df.groupby("ticker").apply(lambda df: TA.SMM(df, 41, "close"))
