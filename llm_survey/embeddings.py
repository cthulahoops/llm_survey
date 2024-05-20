import json

import click
import numpy as np

from llm_survey.data import load_data
from llm_survey.query import client, sqlite_cache
from llm_survey.templating import template_filter


@sqlite_cache("embedding_cache.db")
def embed_content(content, model="text-embedding-3-small"):
    response = client.embeddings.create(
        model=model,
        input=content,
    )
    return response.data[0].embedding


@click.command()
@click.option("--model", "-m", default="text-embedding-3-small")
def embeddings(model):
    data = load_data("embeddings.jsonl")

    raise ValueError(model)

    for input_file in args.input:
        if input_file.suffix == ".md":
            with input_file.open() as f:
                content = f.read()
            embedding = embed_content(content, model=args.model)
            results.append(
                {
                    "content": content,
                    "model": "human",
                    "embedding": embedding,
                }
            )

    with args.output.open("w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


@template_filter()
def similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def consistency_measure(model_outputs):
    output_sum = sum(output.embedding for output in model_outputs)
    similarities = [
        similarity(output_sum, output.embedding) for output in model_outputs
    ]
    return sum(similarities) / len(similarities)


def consistency_grid(model_outputs):
    results = {}
    for i, output in enumerate(model_outputs):
        for j, output2 in enumerate(model_outputs):
            results[(i, j)] = similarity(output.embedding, output2.embedding)
    return results
