SOURCES=playtest pt_blackjack examples

# Run default code formatter
black:
	black $(SOURCES)

check: test mypy

test:
	pytest

mypy:
	mypy $(SOURCES) --ignore-missing-imports
