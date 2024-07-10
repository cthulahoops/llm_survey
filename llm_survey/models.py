import click

from llm_survey.data import SurveyDb
from llm_survey.query import get_models

IGNORED_MODELS = [
    "openrouter/auto",  # Just a helper for calling other models
    "openrouter/flavor-of-the-week"  # Get a random model.
    "liuhaotian/llava-13b",  # Requires an image.
    "meta-llama/llama-3-8b",  # Timeouts
]


@click.command
def models():
    survey = SurveyDb()
    survey.create_tables()

    models = get_models()
    for model in sorted(models, key=lambda x: x.id):
        if model.id in IGNORED_MODELS:
            continue
        survey.save_model(model)
