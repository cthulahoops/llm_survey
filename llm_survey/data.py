import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict

import marshmallow
import numpy as np


@dataclass
class ModelOutput:
    content: str
    embedding: np.array
    model: str
    usage: Dict = None
    evaluation: str = None

    @property
    def score(self):
        if not self.evaluation:
            return None

        if match := re.search(r'"score":\s*([0-9.]+)', self.evaluation):
            return float(match.group(1))


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
    evaluation = marshmallow.fields.Str(required=False, allow_none=True)

    @marshmallow.post_load
    def make_model_output(self, data, **kwargs):
        return ModelOutput(**data)


def load_data(filename):
    with open(filename) as f:
        for line in f:
            data = json.loads(line)
            schema = ModelOutputSchema()
            yield schema.load(data)


def save_data(filename, data):
    schema = ModelOutputSchema()
    with open(filename, "w") as f:
        for item in data:
            f.write(json.dumps(schema.dump(item)) + "\n")


def groupby(data, key):
    result = defaultdict(list)
    for item in data:
        result[key(item)].append(item)
    return result
