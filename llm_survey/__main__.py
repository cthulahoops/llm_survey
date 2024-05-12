import click

from .build import build
from .evaluate import evaluate
from .run import run


@click.group()
def cli():
    pass


cli.add_command(run)
cli.add_command(evaluate)
cli.add_command(build)

if __name__ == "__main__":
    cli()
