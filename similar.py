import json
import numpy as np


from llm_survey.templating import environment
from llm_survey.data import load_data, groupby


def similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def generate_similarity_matrix():
    data = list(load_data())
    data = sorted(data, key=lambda x: x["model"])

    for item in data:
        item["similarities"] = [
            {
                "model": item2["model"],
                "similarity": similarity(item["embedding"], item2["embedding"]),
            }
            for item2 in data
        ]

    return data


def sum_each_model(data):
    grouped = groupby(data, key=lambda x: x.model)

    result = []
    for model, group in grouped.items():
        result.append(
            {"model": model, "embedding": sum(item.embedding for item in group)}
        )
    return result


def main():
    data = list(load_data())
    data = sum_each_model(data)
    data = sorted(data, key=lambda x: x["model"])

    for item in data:
        item["similarities"] = [
            similarity(item["embedding"], item2["embedding"]) for item2 in data
        ]

    template = environment.get_template("similarity.html.j2")

    rendered_html = template.render(data=data)

    with open("out/similarity.html", "w") as outfile:
        outfile.write(rendered_html)


if __name__ == "__main__":
    main()
