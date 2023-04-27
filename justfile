test:
    poetry run nox -s test

benchmark:
    poetry run nox -s benchmark

lint:
    poetry run black .
    poetry run ruff . --fix

pre-commit:
    poetry run pre-commit run --all