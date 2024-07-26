import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import click
from tqdm import tqdm

from llm_survey.data import SurveyDb, groupby, load_data
from llm_survey.embeddings import consistency_grid, consistency_measure, similarity
from llm_survey.templating import (
    get_environment,
    model_company,
    model_file,
    render_to_file,
)

OUTPUT_DIR = Path("out")


@click.command()
@click.option(
    "--pages",
    "-p",
    multiple=True,
    default=["index", "models", "human", "similarity", "consistency", "rankings"],
)
def build(pages):
    survey = SurveyDb()
    environment = get_environment()
    index_template = environment.get_template("index.html.j2")
    template = environment.get_template("model.html.j2")

    data = survey.model_outputs()
    data = groupby(data, key=lambda x: x.model)

    models = sorted(data.keys())

    companies = groupby(models, key=model_company)

    prompt_struct = survey.get_prompt("marshmallow")
    prompt = prompt_struct.prompt

    scores = score_each_model(data)
    costs = average_costs(data)

    rendered_html = index_template.render(
        models=sorted(
            data.keys(),
            key=lambda x: scores[x] or 0,
            reverse=True,
        ),
        prompt=prompt,
        companies=companies,
        scores=scores,
        costs=costs,
        data=data,
    )

    with open("out/index.html", "w") as outfile:
        outfile.write(rendered_html)

    if "models" in pages:
        for model, items in tqdm(list(data.items())):
            rendered_html = template.render(
                items=items,
                current_model=model,
                model_info=survey.get_model(model),
                models=models,
                prompt=prompt,
                companies=companies,
                consistency=consistency_grid(items),
                GRID_SIZE=len(items),
            )

            with (OUTPUT_DIR / model_file(model)).open("w") as outfile:
                outfile.write(rendered_html)

    solutions = list(load_data("solution.jsonl"))
    reference_model = sum_each_model(groupby(solutions, key=lambda x: x.model))["human"]

    render_to_file(
        template,
        "out/human.html",
        items=solutions,
        current_model="human",
        model_info=None,
        models=models,
        prompt=prompt,
        companies=companies,
        consistency=consistency_grid(solutions),
        GRID_SIZE=len(solutions),
    )

    summed_models = sum_each_model(data)
    similarities = similarity_matrix(summed_models)
    consistencies = {model: consistency_measure(items) for model, items in data.items()}

    render_to_file(
        "similarity.html.j2",
        "out/similarity.html",
        models=models,
        summed_models=summed_models,
    )

    render_to_file(
        "consistency.html.j2",
        "out/consistency.html",
        data=per_model_consistency(data),
        outputs=data,
        GRID_SIZE=3,
    )

    render_to_file(
        "rankings.html.j2",
        "out/rankings.html",
        models=sorted(
            data.keys(),
            key=lambda x: scores[x] or 0,
            reverse=True,
        ),
        similarities=similarities,
        reference_model=reference_model,
        summed_models=summed_models,
        scores=score_each_model(data),
        costs=costs,
        consistencies=consistencies,
        data=data,
    )


def per_model_consistency(data):
    data = sorted(data.items(), key=lambda x: consistency_measure(x[1]), reverse=True)
    return [(model, consistency_grid(items)) for model, items in data]


def average_costs(data):
    result = {}
    for model, items in data.items():
        costs = [item.usage["total_cost"] for item in items]
        cost = sum(Decimal(str(cost)) for cost in costs) / len(costs)
        result[model] = cost
    return result


def sum_each_model(grouped):
    return {
        model: sum(item.embedding for item in group) for model, group in grouped.items()
    }


def score_each_model(grouped):
    return {
        model: (
            sum(item.score for item in group if item.score)
            if any(item.score for item in group)
            else None
        )
        for model, group in grouped.items()
    }


def similarity_matrix(embeddings):
    return {
        (model1, model2): similarity(embedding1, embedding2)
        for model1, embedding1 in embeddings.items()
        for model2, embedding2 in embeddings.items()
    }


def read_model_data():
    result = defaultdict(list)
    with open("llm_log.jsonl") as f:
        for line in f:
            data = json.loads(line)

            if "messages" not in data:
                continue

            last_message = data["messages"][-1]

            if last_message["role"] != "assistant":
                continue

            item = {"model": data["model"], "content": last_message["content"]}
            if len(result[data["model"]]) < 3:
                result[data["model"]].append(item)
    return result
