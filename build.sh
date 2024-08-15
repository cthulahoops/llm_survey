#!/bin/sh

export PATH="/opt/render/project/poetry/bin:$PATH"
unset OPENROUTER_API_KEY
unset OPENAI_API_KEY

poetry install
poetry run python -m llm_survey build marshmallow
