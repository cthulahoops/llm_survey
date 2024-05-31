import os

import click
import numpy as np
import openai

from llm_survey.data import SurveyDb
from llm_survey.query import sqlite_cache
from llm_survey.templating import template_filter

client = openai.Client(
    api_key=os.environ.get("OPENAI_API_KEY"),
)


@sqlite_cache("embedding_cache.db")
def embed_content(content, model="text-embedding-3-small"):
    response = client.embeddings.create(
        model=model,
        input=content,
    )
    return np.array(response.data[0].embedding)


@click.command()
@click.option("--model", "-m", default="text-embedding-3-small")
def embeddings(model):
    survey = SurveyDb()

    for output in survey.model_outputs():
        if output.embedding is not None:
            continue
        print("Generate embedding for: ", output.id, output.model)
        output.embedding = embed_content(output.content, model=model)
        survey.save_model_output(output)


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
