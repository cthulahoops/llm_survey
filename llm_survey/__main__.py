import click

from llm_survey.data import SurveyDb
from llm_survey.query import get_models

from .build import build
from .embeddings import embeddings
from .evaluate import evaluate
from .run import run


@click.group()
def cli():
    pass


cli.add_command(embeddings)
cli.add_command(run)
cli.add_command(evaluate)
cli.add_command(build)


@cli.command
def models():
    survey = SurveyDb()
    survey.create_tables()

    models = get_models()
    for model in sorted(models, key=lambda x: x.id):
        survey.save_model(model)


if __name__ == "__main__":
    cli()
