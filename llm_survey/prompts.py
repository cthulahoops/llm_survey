import sys

import click

from llm_survey.data import SurveyDb


@click.command
@click.argument("prompt_id")
def prompts(prompt_id):
    survey = SurveyDb()
    survey.create_tables()

    prompt_content = sys.stdin.read()

    survey.save_prompt(prompt_id, prompt_content)
