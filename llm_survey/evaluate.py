import click
from tqdm import tqdm

from llm_survey.data import Evaluation, SurveyDb
from llm_survey.query import get_completion, reuse_request_if_possible

# I think this is the model I did the original marking with.
# I originally used "openai/gpt-4-turbo-preview" but that's unstable and has been
# updated to a worse model.
DEFAULT_EVALUATION_MODEL = "openai/gpt-4-0125-preview"

get_or_reuse_completion = reuse_request_if_possible(get_completion)


@click.command()
def evaluate():
    prompt_template = open("evaluation_prompt_template.md").read()

    survey = SurveyDb()

    prompt = survey.get_prompt("marshmallow")

    model_outputs = [
        output for output in survey.model_outputs() if output.evaluation is None
    ]

    evaluation_model = survey.get_model(DEFAULT_EVALUATION_MODEL)

    work = tqdm(model_outputs)
    for model_output in work:
        work.set_description(f"{model_output.model:30}")

        evaluation_prompt = prompt_template.format(
            problem=prompt.prompt,
            marking_scheme=prompt.marking_scheme,
            solution=model_output.content,
        )
        request_id, completion = get_or_reuse_completion(
            survey,
            DEFAULT_EVALUATION_MODEL,
            evaluation_prompt,
        )
        evaluation = Evaluation.from_completion(
            model_output,
            evaluation_model,
            completion,
            request_id,
        )
        survey.insert(evaluation)
