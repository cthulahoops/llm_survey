import click

from llm_survey.data import Model, SurveyDb
from llm_survey.query import get_models

IGNORED_MODELS = [
    "openrouter/auto",  # Just a helper for calling other models
    "openrouter/flavor-of-the-week",  # Get a random model.
    "liuhaotian/llava-13b",  # Requires an image.
    "meta-llama/llama-3-8b",  # Timeouts
]


@click.command
def models():
    survey = SurveyDb()
    survey.create_tables()

    request_id, response = get_models(survey)
    models = response["data"]

    models = [Model.from_openai(model) for model in models]

    for model in sorted(models, key=lambda x: x.id):
        if is_ignored(model.id):
            continue
        survey.insert(model)


def is_ignored(model_id):
    return model_id in IGNORED_MODELS
