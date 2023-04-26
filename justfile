test:
    nox -s test

benchmark:
    nox -s benchmark

lint:
    black .
    ruff . --fix