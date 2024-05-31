import click

from llm_survey.data import load_data, save_data
from llm_survey.query import get_model_response


def get_evaluation(model, prompt, response_to_evaluate):
    return get_model_response(model, prompt + response_to_evaluate)


@click.group()
def evaluate():
    data = load_data("embeddings.jsonl")

    prompt = open("evaluation.md").read()

    data = list(data)

    count = 0
    for item in data:
        print("#", count, " - ", item.model)

        content = get_evaluation("openai/gpt-4-turbo", prompt, item.content)
        item.evaluation = content

        count += 1
        if count % 3 == 0:
            save_data("evaluation.jsonl", data)

    save_data("evaluation.jsonl", data)
