.PHONY: clean data lint requirements

requirements: test_environment
	python3 -m pip install -U pip setuptools wheel
	python3 -m pip install -r requirements.txt

create_environment:
	python3 -m venv .venv
	@echo ">>> New virtualenv created."

test_environment:
	python3 test_environment.py

build_dataset: requirements
	python3 src/data/build_dataset.py data/raw/ data/interim/

build_eda_features: build_dataset
	python3 src/features/build_features.py

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

lint:
	black src