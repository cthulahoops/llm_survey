from decimal import Decimal

import click
from tqdm import tqdm

from llm_survey.data import SurveyDb, groupby
from llm_survey.embeddings import consistency_grid, consistency_measure, similarity
from llm_survey.templating import model_company, model_file, render_to_file


class ModelOutputs:
    def __init__(self, data):
        self.data = dict(data)
        self.summed_models = sum_each_model(self.data)

    def by_similarity(self, model_id):
        return sorted(
            (
                other_model
                for other_model in self.summed_models
                if other_model != model_id
            ),
            key=lambda other: similarity(
                self.summed_models[model_id], self.summed_models[other]
            ),
            reverse=True,
        )

    def model_scores(self, model_id, evaluation_model_id="gpt-4-0125-preview"):
        scores = [output.score(evaluation_model_id) for output in self.data[model_id]]
        return [score for score in scores if score is not None]


@click.command()
@click.option(
    "--pages",
    "-p",
    multiple=True,
    default=["index", "models", "human", "similarity", "consistency", "rankings"],
)
@click.argument("prompt_id", default="marshmallow")
def build(prompt_id, pages):
    survey = SurveyDb()

    prompt_struct = survey.get_prompt_outputs(prompt_id)
    if not prompt_struct:
        raise ValueError(f"Prompt {prompt_id!r} does not exist")

    prompt = prompt_struct.prompt

    data = prompt_struct.model_outputs
    data = groupby(data, key=lambda x: x.model)

    outputs = ModelOutputs(data)

    models = sorted(data.keys())
    companies = groupby(models, key=model_company)
    models = sorted(
        data.keys(),
        key=lambda x: sum(outputs.model_scores(x, prompt_struct.evaluation_model)),
        reverse=True,
    )

    costs = average_costs(data)

    summed_models = outputs.summed_models
    reference_model = summed_models.get("human/human")
    consistencies = {model: consistency_measure(items) for model, items in data.items()}

    evaluation_models = sorted(survey.evaluation_models())

    render_to_file(
        "index.html.j2",
        "index.html",
        models=models,
        prompt=prompt_struct,
        companies=companies,
        costs=costs,
        data=data,
        outputs=outputs,
    )

    render_to_file(
        "evaluators.html.j2",
        "evaluators.html",
        models=models,
        prompt=prompt_struct,
        companies=companies,
        costs=costs,
        data=data,
        outputs=outputs,
        evaluation_models=evaluation_models,
    )

    if "models" in pages:
        for model, items in tqdm(list(data.items())):
            render_to_file(
                "model.html.j2",
                model_file(model),
                items=items,
                current_model=model,
                model_info=survey.get_model(model),
                models=models,
                prompt=prompt,
                companies=companies,
                consistency=consistency_grid(items),
                GRID_SIZE=len(items),
                outputs=outputs,
            )

    render_to_file(
        "similarity.html.j2",
        "similarity.html",
        models=models,
        summed_models=summed_models,
    )

    render_to_file(
        "consistency.html.j2",
        "consistency.html",
        data=per_model_consistency(data),
        outputs=data,
        GRID_SIZE=3,
    )

    render_to_file(
        "rankings.html.j2",
        "rankings.html",
        models=models,
        prompt=prompt_struct,
        reference_model=reference_model,
        summed_models=summed_models,
        costs=costs,
        consistencies=consistencies,
        data=data,
        outputs=outputs,
    )


def per_model_consistency(data):
    data = sorted(data.items(), key=lambda x: consistency_measure(x[1]), reverse=True)
    return [(model, consistency_grid(items)) for model, items in data]


def average_costs(data):
    result = {}
    for model, items in data.items():
        costs = [item.usage["total_cost"] for item in items if item.usage]
        if not costs:
            result[model] = Decimal("0.00")
            continue
        cost = sum(Decimal(str(cost)) for cost in costs) / len(costs)
        result[model] = cost
    return result


def sum_each_model(grouped):
    return {
        model: sum(item.embedding for item in group) for model, group in grouped.items()
    }
