init:
	uv sync --frozen

format:
	ruff format .

check:
	ruff check . --fix