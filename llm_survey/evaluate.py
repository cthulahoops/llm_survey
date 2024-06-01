import click
from tqdm import tqdm

from llm_survey.data import SurveyDb
from llm_survey.query import get_model_response


def get_evaluation(model, prompt, response_to_evaluate):
    return get_model_response(model, prompt + response_to_evaluate)


@click.command()
def evaluate():
    prompt = open("evaluation.md").read()

    survey = SurveyDb()

    model_outputs = [
        output for output in survey.model_outputs() if output.evaluation is None
    ]

    work = tqdm(model_outputs)
    for item in work:
        work.set_description(f"{item.model:30}")

        content = get_evaluation("openai/gpt-4-turbo", prompt, item.content)
        item.evaluation = content
        survey.save_model_output(item)
