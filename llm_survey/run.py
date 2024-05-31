import click
from tqdm import tqdm

from llm_survey.data import ModelOutput, SurveyDb, groupby
from llm_survey.query import get_completion


@click.command()
@click.option("--count", default=3)
def run(count=3):
    survey = SurveyDb()

    outputs = survey.model_outputs()
    grouped_outputs = groupby(outputs, key=lambda x: x.model)

    prompt = survey.get_prompt("marshmallow")

    models_needing_work = [
        (model, n + 1)
        for model in survey.models()
        for n in range(count - len(grouped_outputs[model.id]))
    ]

    it = tqdm(models_needing_work, unit="models", postfix={"model": "", "n": ""})
    for model, n in it:
        it.set_postfix(model=model.id, n=n)
        it.write(f"{model.id} {n}")

        completion = get_completion(model.id, prompt)

        model_output = ModelOutput.from_completion(completion, model)
        survey.save_model_output(model_output)
