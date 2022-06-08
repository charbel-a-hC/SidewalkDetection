.PHONY: help
help:
	@echo "Please use 'make <target>' where <target> is one of"
	@echo ""
	@echo "  env		prepare environment and install required dependencies"
	@echo "  clean		remove all temp files along with docker images and docker-compose networks"
	@echo "  format	reformat code"
	@echo ""
	@echo "Check the Makefile to know exactly what each target is doing."



.PHONY: env
env:
	which poetry | grep . && echo 'poetry installed' || curl -sSL https://install.python-poetry.org | python3.7 -
	poetry --version
	poetry env use python3.7
	$(eval VIRTUAL_ENVS_PATH=$(shell poetry env info --path))
	@echo $(VIRTUAL_ENVS_PATH)
	poetry install
	poetry show

.PHONY: env-docker
env-docker:
	which poetry | grep . && echo 'poetry installed' || curl -sSL https://install.python-poetry.org | python3 -
	poetry --version
	poetry install
	poetry show

.PHONY: format
format:
	poetry run bash scripts/format.sh

.PHONY: clean
clean: # Remove Python file artifacts
	find . -name '*.pyc' -exec rm -rf {} +
	find . -name '*.pyo' -exec rm -rf {} +
	find . -name '*~' -exec rm -rf {} +
	find . -name '__pycache__' -exec rm -fr {} +