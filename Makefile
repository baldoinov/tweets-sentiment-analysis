.PHONY: clean data lint requirements

create_environment:
	python3 -m venv .venv
	source .venv/bin/activate
	@echo ">>> New virtualenv created and activated."


test_environment:
	python3 test_environment.py


requirements: test_environment
	python3 -m pip install -U pip setuptools wheel
	python3 -m pip install -r requirements.txt


build_dataset: requirements
	python3 src/build_dataset.py data/raw/ data/interim/with-emoticons/
	python3 src/build_dataset.py data/raw/ data/interim/without-emoticons/ --emoticons False



clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

lint:
	black src