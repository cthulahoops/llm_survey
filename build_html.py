from jinja2 import Environment, FileSystemLoader, select_autoescape
import markdown
import json
from collections import defaultdict

markdown = markdown.Markdown(extensions=["markdown.extensions.fenced_code"])


def main():
    # Set up Jinja2 environment
    environment = Environment(
        loader=FileSystemLoader("."), autoescape=select_autoescape([])
    )

    index_template = environment.get_template("templates/index.html.j2")

    template = environment.get_template("templates/model.html.j2")

    data = read_model_data()
    models = sorted(model_name(model) for model in data.keys())

    rendered_html = index_template.render(models=models)
    with open("out/index.html", "w") as outfile:
        outfile.write(rendered_html)

    for model, items in data.items():
        items = [markdown.convert(item["content"]) for item in items]
        rendered_html = template.render(items=items, model=model, models=models)

        with open(f"out/{model_name(model)}.html", "w") as outfile:
            outfile.write(rendered_html)


def model_name(model):
    return model.split("/")[-1].replace(":", "_")


def read_model_data():
    result = defaultdict(list)
    with open(".chatcli.log") as f:
        for line in f:
            data = json.loads(line)

            if "messages" not in data:
                continue

            last_message = data["messages"][-1]

            if last_message["role"] != "assistant":
                continue

            item = {"model": data["model"], "content": last_message["content"]}
            result[data["model"]].append(item)
    return result


if __name__ == "__main__":
    main()
