import json
import sys

import click
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString

from llm_survey.data import Prompt, SurveyDb


def LS(s):
    return LiteralScalarString(s)


@click.group
def prompts():
    pass


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
        click.echo(json.dumps(prompt.to_dict()))


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

    survey.insert(Prompt(**prompt))


@prompts.command()
@click.argument("name")
@click.option("--marking-prompt/--prompt", default=False)
def add(name, marking_prompt):
    survey = SurveyDb()
    survey.create_tables()

    prompt = survey.get_prompt(name) or Prompt(id=name)

    prompt_content = sys.stdin.read()

    if marking_prompt:
        prompt.marking_prompt = prompt_content
    else:
        prompt.prompt = prompt_content

    survey.save_prompt(prompt)


@prompts.command()
@click.argument("name")
@click.option("--marking-prompt/--prompt", default=False)
def get(name, marking_prompt=False):
    click.echo(f"Getting prompt {name} {marking_prompt}")
    survey = SurveyDb()
    prompt = survey.get_prompt(name)

    if marking_prompt:
        click.echo(prompt.marking_prompt)
    else:
        click.echo(prompt.prompt)


@prompts.command()
def list():
    survey = SurveyDb()
    prompts = survey.prompts()

    for prompt in prompts:
        click.echo(prompt.id)
