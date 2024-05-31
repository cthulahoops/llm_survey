#!/bin/sh

export PATH="/opt/render/project/poetry/bin:$PATH"

poetry install
poetry run python -m llm_survey build
