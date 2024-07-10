import click

from llm_survey.data import SurveyDb


@click.command
def init():
    survey = SurveyDb()
    survey.create_tables()
