from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
import markdown
import json
from collections import defaultdict

markdown = markdown.Markdown(extensions=["markdown.extensions.fenced_code", "nl2br"])


OUTPUT_DIR = Path("out")
TEMPLATE_DIR = Path("templates")


def main():
    # Set up Jinja2 environment
    environment = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR), autoescape=select_autoescape([])
    )

    index_template = environment.get_template("index.html.j2")

    template = environment.get_template("model.html.j2")

    data = read_model_data()
    models = sorted(
        (
            {
                "name": model_name(model),
                "file": model_file(model),
                "model_name": model_name(model).split("/", 1)[1],
            }
            for model in data.keys()
        ),
        key=lambda x: x["name"],
    )

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
        items = [markdown.convert(item["content"]) for item in items]
        rendered_html = template.render(
            items=items,
            model_name=model_name(model),
            models=models,
            prompt=prompt,
            companies=companies,
        )

        with (OUTPUT_DIR / model_file(model)).open("w") as outfile:
            outfile.write(rendered_html)


def groupby(data, key):
    result = defaultdict(list)
    for item in data:
        result[key(item)].append(item)
    return result


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
