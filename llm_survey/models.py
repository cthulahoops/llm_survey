import click

from llm_survey.data import SurveyDb
from llm_survey.query import get_models


@click.command
def models():
    survey = SurveyDb()
    survey.create_tables()

    models = get_models()
    for model in sorted(models, key=lambda x: x.id):
        survey.save_model(model)
