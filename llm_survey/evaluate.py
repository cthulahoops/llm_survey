import click
from tqdm import tqdm

from llm_survey.data import Evaluation, SurveyDb
from llm_survey.query import get_completion, reuse_request_if_possible

# I think this is the model I did the original marking with.
# I originally used "openai/gpt-4-turbo-preview" but that's unstable and has been
# updated to a worse model.
DEFAULT_EVALUATION_MODEL = "openai/gpt-4-1106-preview"

get_or_reuse_completion = reuse_request_if_possible(get_completion)


@click.command()
@click.option("--dry-run", "-n", is_flag=True)
@click.option("--limit", "-l", type=int)
@click.option("--match", "-m")
@click.option("--evaluation-model", "-e")
@click.argument("prompt_id")
def evaluate(prompt_id, dry_run=False, limit=None, match=None, evaluation_model=None):
    prompt_template = open("evaluation_prompt_template.md").read()

    survey = SurveyDb()

    prompt = survey.get_prompt_outputs(prompt_id)

    model_outputs_needing_evaluation = [
        output for output in prompt.model_outputs if output.evaluation is None
    ]

    if match:
        model_outputs_needing_evaluation = [
            output
            for output in model_outputs_needing_evaluation
            if match in output.model
        ]

    if limit:
        model_outputs_needing_evaluation = model_outputs_needing_evaluation[:limit]

    evaluation_model = survey.get_model(evaluation_model or prompt.evaluation_model)
    assert evaluation_model is not None, "No evaluation model configured."

    work = tqdm(model_outputs_needing_evaluation)
    for model_output in work:
        work.set_description(f"{model_output.model:30}")

        if dry_run:
            work.write(f"{model_output.model}")
            continue

        evaluation_prompt = prompt_template.format(
            problem=prompt.prompt,
            marking_scheme=prompt.marking_scheme,
            solution=model_output.content,
        )
        request_id, completion = get_or_reuse_completion(
            survey,
            evaluation_model.id,
            evaluation_prompt,
        )
        evaluation = Evaluation.from_completion(
            model_output,
            evaluation_model,
            completion,
            request_id,
        )
        survey.insert(evaluation)
