"""Functionality for the indicator module.

This module contains all the functions used to calculate the
various technical indicators supported.
"""
import polars as pl

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


def _apply_expr(
    ohlcv_df: pl.LazyFrame,
    ohlcv_columns: list[str],
    expr: pl.Expr,
    identifier_column,
    suffix: str,
) -> pl.LazyFrame:
    if identifier_column is None:
        return ohlcv_df.select(
            pl.all().exclude(ohlcv_columns),
            expr.suffix(suffix),
        )
    else:
        return ohlcv_df.select(
            pl.all().exclude(ohlcv_columns),
            expr.over(identifier_column).suffix(suffix),
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
    suffix = f"_sma_{period}"
    columns = _get_ohlcv_columns(ohlc_df)
    expr = pl.col(columns).rolling_mean(period)
    return _apply_expr(ohlc_df, columns, expr, identifier_column, suffix)


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
