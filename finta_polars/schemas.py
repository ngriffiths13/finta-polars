"""Module containing schema validation code for the finta_polars package."""

import polars as pl

INDICATOR_SCHEMA = {
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
}
INDICATOR_VOLUME_SCHEMA = {
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}

INDICATOR_VOLUME_SCHEMA_INT = INDICATOR_VOLUME_SCHEMA.copy()
INDICATOR_VOLUME_SCHEMA_INT["volume"] = pl.Int64


class PolarsSchemaError(Exception):
    """Exception raised when a polars schema is invalid."""

    pass


def validate_indicator_schema(df: pl.LazyFrame, include_volume: bool = False) -> None:
    """Validate that the schema of a dataframe matches the expected schema.

    Args:
    df (pl.LazyFrame): Dataframe to validate.
    include_volume (bool, optional): Whether to include volume in the schema.
        Defaults to False.

    Raises:
    PolarsSchemaError: If the schema of the dataframe does not match the
        expected schema.
    """
    schema = dict(df.schema)
    if include_volume:
        expected_schemas = [INDICATOR_VOLUME_SCHEMA, INDICATOR_VOLUME_SCHEMA_INT]
    else:
        expected_schemas = [INDICATOR_SCHEMA]
    if not _check_schemas(schema, expected_schemas):
        raise PolarsSchemaError(
            f"""
            Schema of dataframe does not match expected schema.
            Expected one of: {expected_schemas},
            Actual: {schema}
            """
        )


def _check_schemas(schema: dict, expected_schemas: list[dict]) -> bool:
    """Check that the schema of a dataframe matches the expected schema.

    Args:
    schema (dict): Schema to validate.
    expected_schemas (list[dict]): List of acceptable schema.

    Raises:
    PolarsSchemaError: If the schema of the dataframe does not match the
        expected schema.
    """
    good_schema = False
    for expected_schema in expected_schemas:
        if set(expected_schema.items()).issubset(set(schema.items())):
            good_schema = True
            break
    return good_schema
