"""Functionality for the indicator module.

This module contains all the functions used to calculate the
various technical indicators supported.
"""
from typing import Callable

import polars as pl

from finta_polars.expressions import ma_expr
from finta_polars.schemas import validate_indicator_schema

OHLC_COLUMNS = ["open", "high", "low", "close"]
OHLCV_COLUMNS = [*OHLC_COLUMNS, "volume"]


def make_lazy(func):
    """Decorator to make a dataframe lazy before computation.

    This should be over every function that takes a dataframe as input.
    This assumes the DataFrame is always the first argument of a function.
    """

    def wrapper(*args, **kwargs):
        if len(args) > 0:
            if isinstance(args[0], pl.DataFrame):
                args = list(args)
                args[0] = args[0].lazy()

        if "ohlc_df" in kwargs:
            kwargs["ohlc_df"] = kwargs["ohlc_df"].lazy()

        if "ohlcv_df" in kwargs:
            kwargs["ohlcv_df"] = kwargs["ohlcv_df"].lazy()

        return func(*args, **kwargs)

    return wrapper


def _get_ohlcv_columns(ohlc_df: pl.LazyFrame) -> list[str]:
    """Get the OHLCV columns from a dataframe.

    Args:
        ohlc_df (pl.DataFrame): Dataframe containing the OHLCV data.

    Returns:
        list[str]: List of OHLCV columns.
    """
    if "volume" in ohlc_df.columns:
        columns = OHLCV_COLUMNS
        validate_indicator_schema(ohlc_df, include_volume=True)
    else:
        columns = OHLC_COLUMNS
        validate_indicator_schema(ohlc_df, include_volume=False)
    return columns


def _add_identifier_over_to_expr(
    expr: pl.Expr | list[pl.Expr],
    identifier_column: str | None,
) -> list[pl.Expr]:
    """Add an identifier column to an expression."""
    if not isinstance(expr, list):
        expr = [expr]
    if identifier_column is not None:
        expr = [e.over(identifier_column) for e in expr]
    return expr


def _apply_expr(
    ohlcv_df: pl.LazyFrame,
    ohlcv_columns: list[str],
    expr: pl.Expr | list[pl.Expr],
    identifier_column,
    suffix: str,
) -> pl.LazyFrame:
    """Apply an expression to a dataframe."""
    expr = _add_identifier_over_to_expr(expr, identifier_column)
    expr = [e.suffix(suffix) for e in expr]

    return ohlcv_df.select(
        pl.all().exclude(ohlcv_columns),
        *expr,
    )


@make_lazy
def simple_moving_average(
    ohlc_df: pl.LazyFrame,
    period: int = 20,
    identifier_column: str | None = None,
) -> pl.LazyFrame:
    """Calculates the moving average of a dataframe.

    This requires the DataFrame to already be sorted upon calling this function.

    Args:
        ohlc_df (pl.LazyFrame): Dataframe containing the OHLC data.
            Volume can optionally be included.
        period (int, optional): Period to use for the moving average.
            Defaults to 20.
        identifier_column (str, optional): Column to use as an identifier of instrument
            in the dataframe. Defaults to None. If None, the dataframe is assumed to
            contain data for only one instrument.

    Returns:
        pl.LazyFrame: Dataframe containing the moving average of all OHLCV columns.
            Other columns are returned as they were given.
            This makes it convenient to join commands.
    """
    columns = _get_ohlcv_columns(ohlc_df)
    return ohlc_df.select(
        ma_expr(columns, period, None, identifier_column, f"_sma_{period}")
    )


@make_lazy
def simple_moving_median(
    ohlc_df: pl.LazyFrame,
    period: int = 20,
    identifier_column: str | None = None,
) -> pl.LazyFrame:
    """Calculates the moving median of a dataframe.

    This requires the DataFrame to already be sorted upon calling this function.

    Args:
        ohlc_df (pl.LazyFrame): Dataframe containing the OHLC data.
            Volume can optionally be included.
        period (int, optional): Period to use for the moving median.
            Defaults to 20.
        identifier_column (str, optional): Column to use as an identifier of instrument
            in the dataframe. Defaults to None. If None, the dataframe is assumed to
            contain data for only one instrument.

    Returns:
        pl.LazyFrame: Dataframe containing the moving median of all OHLCV columns.
            Other columns are returned as they were given.
            This makes it convenient to join commands.
    """
    suffix = f"_smm_{period}"
    columns = _get_ohlcv_columns(ohlc_df)
    expr = pl.col(columns).rolling_median(period)
    return _apply_expr(ohlc_df, columns, expr, identifier_column, suffix)


@make_lazy
def moving_std(
    ohlc_df: pl.LazyFrame,
    period: int = 20,
    identifier_column: str | None = None,
) -> pl.LazyFrame:
    """Calculates the moving std of a dataframe.

    This requires the DataFrame to already be sorted upon calling this function.

    Args:
        ohlc_df (pl.LazyFrame): Dataframe containing the OHLC data.
            Volume can optionally be included.
        period (int, optional): Period to use for the moving std.
            Defaults to 20.
        identifier_column (str, optional): Column to use as an identifier of instrument
            in the dataframe. Defaults to None. If None, the dataframe is assumed to
            contain data for only one instrument.

    Returns:
        pl.LazyFrame: Dataframe containing the moving std of all OHLCV columns.
            Other columns are returned as they were given.
            This makes it convenient to join commands.
    """
    suffix = f"_msd_{period}"
    columns = _get_ohlcv_columns(ohlc_df)
    expr = pl.col(columns).rolling_std(period)
    return _apply_expr(ohlc_df, columns, expr, identifier_column, suffix)


@make_lazy
def exponential_moving_average(
    ohlc_df: pl.LazyFrame,
    period: int = 20,
    identifier_column: str | None = None,
) -> pl.LazyFrame:
    """Calculates the exponential moving average of a dataframe.

    This requires the DataFrame to already be sorted upon calling this function.

    Args:
        ohlc_df (pl.LazyFrame): Dataframe containing the OHLC data.
            Volume can optionally be included.
        period (int, optional): Period to use for the exponential moving average.
            Defaults to 20.
        identifier_column (str, optional): Column to use as an identifier of instrument
            in the dataframe. Defaults to None. If None, the dataframe is assumed to
            contain data for only one instrument.

    Returns:
        pl.LazyFrame: Dataframe containing the exponential moving average of all OHLCV
            columns. Other columns are returned as they were given.
            This makes it convenient to join commands.
    """
    suffix = f"_ema_{period}"
    columns = _get_ohlcv_columns(ohlc_df)
    expr = pl.col(columns).ewm_mean(span=period)
    return _apply_expr(ohlc_df, columns, expr, identifier_column, suffix)


@make_lazy
def typical_price(
    ohlc_df: pl.LazyFrame,
) -> pl.LazyFrame:
    """Calculate the typical price defined as the arithmetic mean of high low and close.

    Args:
        ohlc_df (pl.LazyFrame): Dataframe containing the OHLC data.

    Returns:
        pl.LazyFrame: Dataframe containing the typical price.
            Other columns are returned as they were given.
            This makes it convenient to join commands.
    """
    tp = ohlc_df.with_columns(
        ((pl.col("high") + pl.col("low") + pl.col("close")) / 3).alias("typical_price")
    )
    return tp


@make_lazy
def vwap(
    ohlcv_df: pl.LazyFrame,
) -> pl.LazyFrame:
    """Calculates the volume weighted average price.

    Args:
        ohlcv_df (pl.LazyFrame): Dataframe containing the OHLCV data.

    Returns:
        pl.LazyFrame: Dataframe containing the volume weighted average price.
            Other columns are returned as they were given.
            This makes it convenient to join commands.
    """
    ...


@make_lazy
def rsi(
    ohlc_df: pl.LazyFrame,
    period: int = 14,
    identifier_column: str | None = None,
) -> pl.LazyFrame:
    """Calculates the relative strength index.

    Args:
        ohlc_df (pl.LazyFrame): Dataframe containing the OHLC data.
            Volume can optionally be included.
        period (int, optional): Period to use for the relative strength index.
            Defaults to 14.
        identifier_column (str, optional): Column to use as an identifier of instrument
            in the dataframe. Defaults to None. If None, the dataframe is assumed to
            contain data for only one instrument.

    Returns:
        pl.LazyFrame: Dataframe containing the relative strength index.
            Other columns are returned as they were given.
            This makes it convenient to join commands.
    """
    ...


@make_lazy
def bbands(
    ohlc_df: pl.LazyFrame,
    period: int = 20,
    std: float = 2.0,
    identifier_column: str | None = None,
    ma_func: Callable[[pl.Series], pl.Series] = simple_moving_average,
) -> pl.LazyFrame:
    """Calculates the bollinger bands.

    Args:
        ohlc_df (pl.LazyFrame): Dataframe containing the OHLC data.
            Volume can optionally be included.
        period (int, optional): Period to use for the bollinger bands.
            Defaults to 20.
        std (float, optional): Standard deviation to use for the bollinger bands.
            Defaults to 2.0.
        identifier_column (str, optional): Column to use as an identifier of instrument
            in the dataframe. Defaults to None. If None, the dataframe is assumed to
            contain data for only one instrument.
        ma_func (Callable[[pl.Series], pl.Series], optional): Function to use for
            calculating the moving average. Defaults to simple_moving_average.

    Returns:
        pl.LazyFrame: Dataframe containing the bollinger bands.
            Other columns are returned as they were given.
            This makes it convenient to join commands.
    """
    ...


@make_lazy
def macd(
    ohlc_df: pl.LazyFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    identifier_column: str | None = None,
) -> pl.LazyFrame:
    """Calculates the moving average convergence divergence.

    Args:
        ohlc_df (pl.LazyFrame): Dataframe containing the OHLC data.
            Volume can optionally be included.
        fast_period (int, optional): Period to use for the fast moving average.
            Defaults to 12.
        slow_period (int, optional): Period to use for the slow moving average.
            Defaults to 26.
        signal_period (int, optional): Period to use for the signal line.
            Defaults to 9.
        identifier_column (str, optional): Column to use as an identifier of instrument
            in the dataframe. Defaults to None. If None, the dataframe is assumed to
            contain data for only one instrument.

    Returns:
        pl.LazyFrame: Dataframe containing the moving average convergence divergence.
            Other columns are returned as they were given.
            This makes it convenient to join commands.
    """
    suffix = f"_macd_{fast_period}_{slow_period}"
    columns = OHLC_COLUMNS
    expr = [
        pl.col(col).ewm_mean(span=slow_period) - pl.col(col).ewm_mean(span=fast_period)
        for col in columns
    ]
    out = _apply_expr(ohlc_df, columns, expr, identifier_column, suffix)
    expr = [pl.col(c + suffix).ewm_mean(span=signal_period) for c in columns]
    expr = _add_identifier_over_to_expr(expr, identifier_column)
    expr = [e.suffix("_signal") for e in expr]
    return out.with_columns(expr)
