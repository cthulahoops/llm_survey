import json
import sys

import click
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString

from llm_survey.data import Prompt, SurveyDb
from llm_survey.evaluate import DEFAULT_EVALUATION_MODEL


def LS(s):
    return LiteralScalarString(s)


@click.group
def prompts():
    pass


@prompts.command()
@click.argument("name")
def new(name):
    survey = SurveyDb()
    survey.create_tables()

    survey.insert(Prompt(id=name, evaluation_model=DEFAULT_EVALUATION_MODEL))


@prompts.command()
@click.argument("name")
@click.option("--format", default="json", type=click.Choice(["json", "yaml"]))
def export(name, format):
    survey = SurveyDb()

    prompt = survey.get_prompt(name)
    if format == "yaml":
        yaml = YAML()
        output = prompt.to_dict()
        output["prompt"] = LS(output["prompt"])
        output["marking_scheme"] = LS(output["marking_scheme"])
        click.echo(yaml.dump(output, sys.stdout))
    else:
        click.echo(json.dumps(prompt.to_dict(), indent=2))


@prompts.command(name="import")
@click.option("--format", default="json", type=click.Choice(["json", "yaml"]))
def import_prompt(format):
    survey = SurveyDb()
    survey.create_tables()

    if format == "yaml":
        yaml = YAML()
        prompt = yaml.load(sys.stdin)
    else:
        prompt = json.load(sys.stdin)

    survey.save_prompt(prompt)


@prompts.command()
@click.argument("name")
def delete(name):
    survey = SurveyDb()

    survey.delete_prompt(name)


@prompts.command()
def list():
    survey = SurveyDb()
    prompts = survey.prompts()

    for prompt in prompts:
        click.echo(prompt.id)
