import click
from tqdm import tqdm

from llm_survey.data import ModelOutput, SurveyDb, groupby
from llm_survey.models import is_ignored
from llm_survey.query import get_completion


@click.command()
@click.option("--count", default=3)
@click.option("--dry-run", "-n", is_flag=True)
def run(dry_run=False, count=3):
    import openai

    survey = SurveyDb()

    outputs = survey.model_outputs()
    grouped_outputs = groupby(outputs, key=lambda x: x.model)

    prompt = survey.get_prompt("marshmallow")

    models_needing_work = [
        (model, n + 1)
        for model in survey.models()
        for n in range(count - len(grouped_outputs[model.id]))
        if not is_ignored(model.id)
    ]

    it = tqdm(models_needing_work, unit="models", postfix={"model": "", "n": ""})
    for model, n in it:
        it.set_postfix(model=model.id, n=n)
        it.write(f"{model.id} {n}")

        if dry_run:
            continue

        try:
            request_id, completion = get_completion(survey, model.id, prompt.prompt)
        except openai.NotFoundError as exc:
            print(exc)
            continue

        if hasattr(completion, "error"):
            print(completion.error)
            continue

        model_output = ModelOutput.from_completion(completion, model, request_id)
        survey.insert(model_output)
