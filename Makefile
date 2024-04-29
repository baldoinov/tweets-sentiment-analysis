.ONESHELL:
.PHONY: clean data lint requirements

ensure_environment:
	find . -type d -name ".venv" -exec rm -rf {} \;
	python3 -m venv .venv
	
	@echo ">>> New virtualenv created. ACTIVATE IT BEFORE PROCEEDING."


test_environment:
	python3 test_environment.py


requirements: test_environment
	python3 -m pip install -e .
	python3 -m pip install pip-tools
	python3 -m pip-compile requirements.in
	python3 -m pip install -r requirements.txt


build_dataset: requirements
	python3 src/build_dataset.py data/raw/ data/interim/with-emoticons/
	python3 src/build_dataset.py data/raw/ data/interim/without-emoticons/ --emoticons False


clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} \;


lint:
	black src