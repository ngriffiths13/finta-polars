import pytest
from finta import TA

from finta_polars.indicators import OHLC_COLUMNS, typical_price


@pytest.mark.benchmark(group="tp")
def test_typical_price_all_prices_polars(ohlcv_df, benchmark):
    """Test that the typical price is correct."""
    tp = typical_price(ohlcv_df)
    benchmark(tp.collect)


@pytest.mark.benchmark(group="tp_multiple_companies")
def test_typical_price_all_prices_multiple_companies_polars(
    ohlcv_df_multiple_companies, benchmark
):
    """Test that the typical price is correct."""
    tp = typical_price(
        ohlcv_df_multiple_companies)
    benchmark(tp.collect)


@pytest.mark.benchmark(group="tp")
def test_typical_price_close_finta(ohlcv_df, benchmark):
    """Test that the typical price is correct."""
    ohlc_df = ohlcv_df.to_pandas()
    benchmark(TA.TP, ohlc_df)


@pytest.mark.benchmark(group="tp")
def test_typical_price_all_prices_naive_finta(ohlcv_df, benchmark):
    """Test that the typical price is correct."""
    ohlc_df = ohlcv_df.to_pandas()

    @benchmark
    def result():
        for c in OHLC_COLUMNS:
            TA.TP(ohlc_df)


@pytest.mark.benchmark(group="tp_multiple_companies")
def test_typical_price_close_multiple_companies_finta(
    ohlcv_df_multiple_companies, benchmark
):
    """Test that the typical price is correct."""
    ohlc_df = ohlcv_df_multiple_companies.to_pandas()

    @benchmark
    def result():
        ohlc_df.groupby("ticker").apply(lambda df: TA.TP(df))
