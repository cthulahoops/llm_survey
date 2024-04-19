import json
import numpy as np


from jinja2 import Environment, FileSystemLoader, select_autoescape


def load_data():
    for line in open("embeddings.jsonl"):
        data = json.loads(line)
        data["embedding"] = np.array(data["embedding"])
        yield data


def similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def generate_similarity_matrix():
    data = list(load_data())
    data = sorted(data, key=lambda x: x["model"])

    for item in data:
        item["similarities"] = [
            {
                "model2": item2["model"],
                "similarity": similarity(item["embedding"], item2["embedding"]),
            }
            for item2 in data
        ]

    return data


def main():
    data = list(load_data())
    data = sorted(data, key=lambda x: x["model"])

    for item in data:
        item["similarities"] = [
            similarity(item["embedding"], item2["embedding"]) for item2 in data
        ]

    environment = Environment(
        loader=FileSystemLoader("."), autoescape=select_autoescape([])
    )

    template = environment.get_template("templates/similarity.html.j2")

    rendered_html = template.render(data=data)

    with open("out/similarity.html", "w") as outfile:
        outfile.write(rendered_html)


if __name__ == "__main__":
    main()
