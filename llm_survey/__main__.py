import click

from .build import build
from .embeddings import embeddings
from .evaluate import evaluate
from .init import init
from .models import models
from .prompts import prompts
from .run import run


@click.group()
def cli():
    pass


cli.add_command(init)
cli.add_command(embeddings)
cli.add_command(run)
cli.add_command(evaluate)
cli.add_command(build)
cli.add_command(models)
cli.add_command(prompts)


if __name__ == "__main__":
    cli()
