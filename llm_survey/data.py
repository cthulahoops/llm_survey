import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict

import marshmallow
import numpy as np


@dataclass
class Model:
    id: str
    name: str
    description: str
    context_length: int
    pricing: Dict

    @classmethod
    def from_row(cls, row):
        return cls(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            context_length=row["context_length"],
            pricing=json.loads(row["pricing"]),
        )


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

    @classmethod
    def from_row(cls, row):
        return cls(
            id=row["id"],
            content=row["content"],
            embedding=np.frombuffer(row["embedding"]) if row["embedding"] else None,
            model=row["model"],
            usage=json.loads(row["usage"]),
            evaluation=row["evaluation"],
        )

    @classmethod
    def from_completion(cls, completion, model):
        assert model.id == completion.model

        message = completion.choices[0].message
        usage = completion.usage
        pricing = model.pricing

        return cls(
            id=None,
            content=message.content,
            model=completion.model,
            usage={
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.completion_tokens,
                "total_cost": (
                    usage.prompt_tokens * Decimal(pricing["prompt"])
                    + usage.completion_tokens * Decimal(pricing["completion"])
                ),
            },
        )


class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        return super().default(obj)


class SurveyDb:
    def __init__(self):
        self.sqlite = sqlite3.connect("survey.db")

    def query(self, query, *args):
        cursor = self.sqlite.execute(query, args)
        cursor.row_factory = sqlite3.Row
        return cursor

    def create_tables(self):
        self.sqlite.execute(
            """CREATE TABLE IF NOT EXISTS models (
            id TEXT PRIMARY KEY,
            name TEXT,
            description TEXT,
            context_length INTEGER,
            pricing TEXT
        )"""
        )

        self.sqlite.execute(
            """CREATE TABLE IF NOT EXISTS prompts (
            id TEXT PRIMARY KEY,
            content TEXT
        )"""
        )

        self.sqlite.execute(
            """CREATE TABLE IF NOT EXISTS model_outputs (
            id INTEGER PRIMARY KEY,
            content TEXT,
            embedding BLOB,
            model TEXT,
            usage TEXT,
            evaluation TEXT
        )"""
        )

        self.sqlite.commit()

    def save_model(self, model):
        self.sqlite.execute(
            """INSERT INTO models VALUES (?, ?, ?, ?, ?)
            on conflict(id) do update set
            name = excluded.name,
            description = excluded.description,
            context_length = excluded.context_length,
            pricing = excluded.pricing
            """,
            (
                model.id,
                model.name,
                model.description,
                model.context_length,
                json.dumps(model.pricing),
            ),
        )
        self.sqlite.commit()

    def save_prompt(self, prompt_id, content):
        self.sqlite.execute(
            """INSERT INTO prompts VALUES (?, ?)
            on conflict(id) do update set
            content = excluded.content
            """,
            (prompt_id, content),
        )
        self.sqlite.commit()

    def save_model_output(self, model_output):
        self.sqlite.execute(
            """INSERT INTO model_outputs
            (content, embedding, model, usage, evaluation)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                model_output.content,
                model_output.embedding,
                model_output.model,
                json.dumps(model_output.usage),
                model_output.evaluation,
            ),
        )
        self.sqlite.commit()

    @property
    def models(self):
        for row in self.query("SELECT * FROM models"):
            yield Model.from_row(row)

    def get_model(self, model_id):
        cursor = self.query("SELECT * FROM models WHERE id=?", model_id)
        row = cursor.fetchone()
        if row is None:
            return None

        return Model.from_row(row)

    def get_prompt(self, prompt_id):
        cursor = self.query("SELECT * FROM prompts WHERE id=?", prompt_id)
        row = cursor.fetchone()
        if row is None:
            return None

        return row["content"]

    def model_outputs(self):
        for row in self.query("SELECT * FROM model_outputs"):
            yield ModelOutput.from_row(row)


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
