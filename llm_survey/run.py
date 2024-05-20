import click

from llm_survey.data import SurveyDb, load_data


@click.command()
def run():
    outputs = load_data("evaluation.jsonl")

    survey = SurveyDb()

    for output in outputs:
        output.model = output.model[len("openrouter/"):]
        survey.save_model_output(output)
    # prompt = open("prompt.md").read()

    # for model in models:
    #     for _ in range(3 - len(data[model])):
    #         completion = get_completion(model, prompt)

    #         model_output = ModelOutput(completion=completion, model=model)

    #         data.append(model_output)

    # save_data("evaluation.jsonl", data)
