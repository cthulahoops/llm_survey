import click

from llm_survey.data import Model, SurveyDb
from llm_survey.query import get_models

IGNORED_MODELS = [
    "openrouter/auto",  # Just a helper for calling other models
    "openrouter/flavor-of-the-week",  # Get a random model.
    "liuhaotian/llava-13b",  # Requires an image.
    "meta-llama/llama-3-8b",  # Timeouts
]


@click.group
def models():
    pass


@models.command
def fetch():
    survey = SurveyDb()
    survey.create_tables()

    existing_models = {model.id for model in survey.models()}

    request_id, response = get_models(survey)
    models = response["data"]

    models_to_add = [Model.from_openai(model) for model in models]

    for model in sorted(models_to_add, key=lambda x: x.id):
        if model.id in existing_models or is_ignored(model.id):
            continue
        survey.insert(model)


def is_ignored(model_id):
    return model_id in IGNORED_MODELS
