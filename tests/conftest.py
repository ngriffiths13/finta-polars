import polars as pl
import pytest


@pytest.fixture
def ohlcv_df():
    return pl.DataFrame(
        {
            "open": [float(i) for i in range(3000)],
            "high": [float(i) + 1 for i in range(3000)],
            "low": [float(i) - 1 for i in range(3000)],
            "close": [float(i) for i in range(3000)],
            "volume": [float(i) for i in range(3000)],
        }
    )


@pytest.fixture
def ohlcv_df_multiple_companies():
    return pl.DataFrame(
        {
            "open": [float(i) for i in range(3000)] * 5,
            "high": [float(i) + 1 for i in range(3000)] * 5,
            "low": [float(i) - 1 for i in range(3000)] * 5,
            "close": [float(i) for i in range(3000)] * 5,
            "volume": [float(i) for i in range(3000)] * 5,
            "ticker": ["AAPL"] * 3000
            + ["MSFT"] * 3000
            + ["GOOG"] * 3000
            + ["AMZN"] * 3000
            + ["FB"] * 3000,
        }
    )
