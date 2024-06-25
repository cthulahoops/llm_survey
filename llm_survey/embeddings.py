import os

import click
from tqdm import tqdm

from llm_survey.data import SurveyDb
from llm_survey.query import sqlite_cache
from llm_survey.templating import template_filter


@click.command()
@click.option("--model", "-m", default="text-embedding-3-small")
def embeddings(model):
    import numpy as np

    survey = SurveyDb()

    it = tqdm(list(survey.model_outputs()), unit="outputs")
    for output in it:
        it.set_postfix(model=output.model, id=output.id)
        if output.embedding is not None:
            continue
        it.write(f"Generate embedding for: {output.id} {output.model}")
        output.embedding = embed_content(output.content, model=model)

        if isinstance(output.embedding, bytes):
            output.embedding = np.frombuffer(output.embedding)

        survey.save_model_output(output)


@sqlite_cache("embedding_cache.db")
def embed_content(content, model="text-embedding-3-small"):
    import numpy as np
    import openai

    client = openai.Client(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    response = client.embeddings.create(
        model=model,
        input=content,
    )
    return np.array(response.data[0].embedding)


@template_filter()
def similarity(a, b):
    import numpy as np

    if a is None or b is None:
        return 0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def consistency_measure(model_outputs):
    if any(output.embedding is None for output in model_outputs):
        return None

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
