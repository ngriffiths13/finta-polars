"""Noxfile for running commands in clean environments."""
from nox import Session, session


@session(python="3.10")
def benchmark(session: Session) -> None:
    """Run benchmarks."""
    session.run("pip", "install", "pytest", "pytest-benchmark", "finta", "pyarrow", ".")
    session.run("pytest", "--benchmark-enable", "-v", "-m", "benchmark")


@session(python="3.10")
def test(session: Session) -> None:
    """Run tests."""
    session.run(
        "pip",
        "install",
        "pytest",
        "pytest-benchmark",
        "pytest-cov",
        "finta",
        "pyarrow",
        ".",
    )
    session.run("pytest", "--cov=finta_polars", "-v", "-m", " not benchmark")
