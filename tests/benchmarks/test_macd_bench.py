import pytest
from finta import TA

from finta_polars.indicators import OHLC_COLUMNS, macd


@pytest.mark.benchmark(group="macd")
def test_simple_moving_std_price_all_prices_polars(ohlcv_df, benchmark):
    """Test that the simple moving average of the close price is correct."""
    macd_out = macd(ohlcv_df)
    benchmark(macd_out.collect)


@pytest.mark.benchmark(group="macd_multiple_companies")
def test_simple_moving_std_price_all_prices_multiple_companies_polars(
    ohlcv_df_multiple_companies, benchmark
):
    """Test that the simple moving average of the close price is correct."""
    macd_out = macd(ohlcv_df_multiple_companies, identifier_column="ticker")
    benchmark(macd_out.collect)


@pytest.mark.benchmark(group="macd")
def test_simple_moving_std_price_close_finta(ohlcv_df, benchmark):
    """Test that the simple moving average of the close price is correct."""
    ohlc_df = ohlcv_df.to_pandas()
    benchmark(TA.MACD, ohlc_df)


@pytest.mark.benchmark(group="macd")
def test_simple_moving_std_price_all_prices_naive_finta(ohlcv_df, benchmark):
    """Test that the simple moving average of the close price is correct."""
    ohlc_df = ohlcv_df.to_pandas()

    @benchmark
    def result():
        for c in OHLC_COLUMNS:
            TA.MACD(ohlc_df, column=c)


@pytest.mark.benchmark(group="macd_multiple_companies")
def test_simple_moving_std_price_close_multiple_companies_finta(
    ohlcv_df_multiple_companies, benchmark
):
    """Test that the simple moving average of the close price is correct."""
    ohlc_df = ohlcv_df_multiple_companies.to_pandas()

    @benchmark
    def result():
        ohlc_df.groupby("ticker").apply(lambda df: TA.MACD(df))
