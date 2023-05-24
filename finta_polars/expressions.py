import polars as pl


def _add_identifier_over_to_expr(
    expr: pl.Expr | list[pl.Expr],
    identifier_columns: list[str] | str | None,
) -> list[pl.Expr]:
    """Add an identifier column to an expression."""
    if not isinstance(expr, list):
        expr = [expr]
    if identifier_columns is not None:
        expr = [e.over(identifier_columns) for e in expr]
    return expr


def ma_expr(
    columns: list[str],
    window_size: int,
    min_periods: int | None = None,
    over_columns: str | list[str] | None = None,
    suffix: str = "",
) -> list[pl.Expr]:
    """Create a moving average expression.

    Args:
        columns (list[str]): Columns to calculate moving average for.
        window_size (int): Window size for moving average.
        min_periods (int, optional): Minimum number of periods to calculate moving average. Defaults to None.
        over_columns (str | list[str], optional): Columns to group by. Defaults to None.
        suffix (str, optional): Suffix to add to column names. Defaults to "".
    """
    expr = [
        pl.col(c).rolling_mean(window_size=window_size, min_periods=min_periods)
        for c in columns
    ]
    expr = _add_identifier_over_to_expr(expr, over_columns)
    expr = [e.suffix(suffix) for e in expr]
    return expr
