#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = src
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python3

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install --no-cache-dir -r requirements.txt


## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml src


## Set up python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "$(PYTHON_INTERPRETER) -m venv .venv"
	@echo ">>> New virtualenv created. Activate it!"
	

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################
