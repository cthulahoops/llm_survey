import click
from tqdm import tqdm

from llm_survey.data import Evaluation, SurveyDb
from llm_survey.query import get_completion

# I think this is the model I did the original marking with.
# I originally used "openai/gpt-4-turbo-preview" but that's unstable and has been
# updated to a worse model.
DEFAULT_EVALUATION_MODEL = "openai/gpt-4-0125-preview"


@click.command()
def evaluate():
    prompt = open("evaluation.md").read()

    survey = SurveyDb()

    model_outputs = [
        output for output in survey.model_outputs() if output.evaluation is None
    ]

    evaluation_model = survey.get_model(DEFAULT_EVALUATION_MODEL)

    work = tqdm(model_outputs)
    for model_output in work:
        work.set_description(f"{model_output.model:30}")

        request_id, completion = get_completion(
            survey,
            DEFAULT_EVALUATION_MODEL,
            prompt + model_output.content,
        )
        evaluation = Evaluation.from_completion(
            model_output,
            evaluation_model,
            completion,
            request_id,
        )
        survey.insert(evaluation)
