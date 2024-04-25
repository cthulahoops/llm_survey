import json
from collections import defaultdict
from pathlib import Path

import markdown

from llm_survey.data import groupby, load_data
from llm_survey.embeddings import consistency_grid, consistency_measure, similarity
from llm_survey.templating import environment, model_company, model_file, render_to_file

OUTPUT_DIR = Path("out")
markdown = markdown.Markdown(extensions=["markdown.extensions.fenced_code", "nl2br"])


def main():
    index_template = environment.get_template("index.html.j2")
    template = environment.get_template("model.html.j2")

    data = load_data("embeddings.jsonl")
    data = groupby(data, key=lambda x: x.model)

    models = sorted(data.keys())

    companies = groupby(models, key=model_company)

    prompt = open("prompt.md").read()
    prompt = markdown.convert(prompt)

    rendered_html = index_template.render(
        models=models,
        prompt=prompt,
        companies=companies,
    )

    with open("out/index.html", "w") as outfile:
        outfile.write(rendered_html)

    for model, items in data.items():
        markdown_items = [markdown.convert(item.content) for item in items]
        rendered_html = template.render(
            items=markdown_items,
            current_model=model,
            models=models,
            prompt=prompt,
            companies=companies,
            consistency=consistency_grid(items),
            GRID_SIZE=3,
        )

        with (OUTPUT_DIR / model_file(model)).open("w") as outfile:
            outfile.write(rendered_html)

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
        GRID_SIZE=3,
    )

    solutions = load_data("solution.json")
    reference_model = next(solutions).embedding

    render_to_file(
        "rankings.html.j2",
        "out/rankings.html",
        models=sorted(
            data.keys(),
            key=lambda x: similarity(summed_models[x], reference_model),
            reverse=True,
        ),
        similarities=similarities,
        reference_model=reference_model,
        summed_models=summed_models,
        costs=average_costs(data),
        consistencies=consistencies,
    )


def per_model_consistency(data):
    data = sorted(data.items(), key=lambda x: consistency_measure(x[1]), reverse=True)
    return [(model, consistency_grid(items)) for model, items in data]


def average_costs(data):
    result = {}
    for model, items in data.items():
        costs = [item.usage["total_cost"] for item in items]
        cost = sum(costs) / len(costs)
        result[model] = cost
    return result


def sum_each_model(grouped):
    return {
        model: sum(item.embedding for item in group) for model, group in grouped.items()
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


if __name__ == "__main__":
    main()
