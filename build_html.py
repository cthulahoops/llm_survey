import json
from collections import defaultdict
from pathlib import Path

import markdown

from llm_survey.data import groupby, load_data
from llm_survey.embeddings import consistency_grid, consistency_measure, similarity
from llm_survey.templating import environment, render_to_file

OUTPUT_DIR = Path("out")
markdown = markdown.Markdown(extensions=["markdown.extensions.fenced_code", "nl2br"])


def main():
    index_template = environment.get_template("index.html.j2")
    template = environment.get_template("model.html.j2")

    data = load_data()
    data = groupby(data, key=lambda x: x.model)

    models = (
        {
            "name": model_name(model),
            "file": model_file(model),
            "model_name": model_name(model).split("/", 1)[1],
        }
        for model in data.keys()
    )

    models = sorted(models, key=lambda x: x["name"])

    companies = groupby(models, key=lambda x: x["name"].split("/")[0])

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
            model_name=model_name(model),
            models=models,
            prompt=prompt,
            companies=companies,
            consistency=consistency_grid(items),
            GRID_SIZE=3,
        )

        with (OUTPUT_DIR / model_file(model)).open("w") as outfile:
            outfile.write(rendered_html)

    costs = average_costs(data)
    similarities = similarity_matrix(data)

    consistency_page(data)
    similarity_page(data, similarities)

    rankings_page(data, similarities, "openrouter/anthropic/claude-3-opus:beta", costs)


def average_costs(data):
    result = {}
    for model, items in data.items():
        costs = [item.usage["total_cost"] for item in items]
        cost = sum(costs) / len(costs)
        result[model] = cost
    return result


def sum_each_model(grouped):
    result = []
    for model, group in grouped.items():
        result.append(
            {"model": model, "embedding": sum(item.embedding for item in group)}
        )
    return result


def similarity_matrix(data):
    data = sum_each_model(data)

    return {
        (item1["model"], item2["model"]): similarity(
            item1["embedding"],
            item2["embedding"],
        )
        for item1 in data
        for item2 in data
    }


def rankings_page(data, similarities, reference_model, costs):
    render_to_file(
        "rankings.html.j2",
        "out/rankings.html",
        models=sorted(
            data.keys(), key=lambda x: similarities[x, reference_model], reverse=True
        ),
        similarities=similarities,
        reference_model=reference_model,
        costs=costs,
    )


def similarity_page(data, similarities):
    render_to_file(
        "similarity.html.j2",
        "out/similarity.html",
        models=sorted(data.keys()),
        similarities=similarities,
    )


def consistency_page(data):
    template = environment.get_template("consistency.html.j2")

    data = sorted(data.items(), key=lambda x: consistency_measure(x[1]), reverse=True)
    grids = [(model, consistency_grid(items)) for model, items in data]

    rendered_html = template.render(data=grids, GRID_SIZE=3)

    with open("out/consistency.html", "w") as outfile:
        outfile.write(rendered_html)


def model_name(model):
    return model[model.find("/") + 1 :]


def model_file(model):
    name = model_name(model)
    return name[name.find("/") + 1 :].replace(":", "_").replace("/", "_") + ".html"


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
