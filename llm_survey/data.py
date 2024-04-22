import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict

import marshmallow
import numpy as np

DATAFILE = "embeddings.jsonl"


@dataclass
class ModelOutput:
    content: str
    embedding: np.array
    model: str
    usage: Dict = None


class NumpyArray(marshmallow.fields.Field):
    def _serialize(self, value, attr, obj, **kwargs):
        return value.tolist()

    def _deserialize(self, value, attr, data, **kwargs):
        return np.array(value)


class ModelOutputSchema(marshmallow.Schema):
    content = marshmallow.fields.Str()
    embedding = NumpyArray()
    model = marshmallow.fields.Str()
    usage = marshmallow.fields.Dict()

    @marshmallow.post_load
    def make_model_output(self, data, **kwargs):
        return ModelOutput(**data)


def load_data():
    with open(DATAFILE) as f:
        for line in f:
            data = json.loads(line)
            schema = ModelOutputSchema()
            yield schema.load(data)


def groupby(data, key):
    result = defaultdict(list)
    for item in data:
        result[key(item)].append(item)
    return result
