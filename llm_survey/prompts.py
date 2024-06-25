import sys

import click

from llm_survey.data import Prompt, SurveyDb


@click.group
def prompts():
    pass


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
