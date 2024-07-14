import click
from tqdm import tqdm

from llm_survey.data import Embedding, SurveyDb
from llm_survey.query import embed_content
from llm_survey.templating import template_filter


@click.command()
@click.option("--model", "-m", default="text-embedding-3-small")
@click.option("--dry-run", "-n", is_flag=True)
def embeddings(model, dry_run=False):
    import numpy as np

    survey = SurveyDb()

    it = tqdm(list(survey.model_outputs()), unit="outputs")
    for output in it:
        it.set_postfix(model=output.model, id=output.id)
        if output.embedding is not None:
            continue
        it.write(f"Generate embedding for: {output.id} {output.model}")

        if dry_run:
            continue

        request_id, embedding = embed_content(survey, output.content, model=model)

        if isinstance(output.embedding, bytes):
            embedding = np.frombuffer(output.embedding)

        survey.insert(
            Embedding(
                output_id=output.id,
                model=model,
                embedding=embedding,
                request_id=request_id,
            )
        )


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
