import numpy as np


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
