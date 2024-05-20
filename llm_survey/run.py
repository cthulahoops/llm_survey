import click

from llm_survey.data import ModelOutput, groupby, load_data, save_data
from llm_survey.query import get_completion


@click.command()
def run():
    models = open("models", "r").read().splitlines()

    data = groupby(load_data("evaluation.jsonl"), lambda x: x.model)

    prompt = open("prompt.md").read()

    for model in models:
        for _ in range(3 - len(data[model])):
            completion = get_completion(model, prompt)

            model_output = ModelOutput(completion=completion, model=model)

            data.append(model_output)

    save_data("evaluation.jsonl", data)
