import json
from collections import defaultdict
from datetime import datetime

import numpy as np
from sqlalchemy import (
    BLOB,
    JSON,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import joinedload, relationship, sessionmaker

Base = declarative_base()


class Model(Base):
    __tablename__ = "models"

    id = Column(String, primary_key=True)
    name = Column(String)
    description = Column(String)
    context_length = Column(Integer)
    pricing = Column(JSON)


class Prompt(Base):
    __tablename__ = "prompts"

    id = Column(String, primary_key=True)
    prompt = Column(String)
    marking_prompt = Column(String)


class ModelOutput(Base):
    __tablename__ = "model_outputs"

    id = Column(Integer, primary_key=True)
    content = Column(String)
    model = Column(String)
    usage = Column(JSON)
    evaluation = Column(String)
    request_id = Column(Integer, ForeignKey("request_logs.id"))

    embeddings = relationship("Embedding", back_populates="model_output")

    @property
    def embedding(self):
        if self.embeddings:
            return np.frombuffer(self.embeddings[0].embedding)
        return None


class Embedding(Base):
    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True)
    output_id = Column(Integer, ForeignKey("model_outputs.id"))
    embedding = Column(BLOB)
    model = Column(String)
    request_id = Column(Integer, ForeignKey("request_logs.id"))

    model_output = relationship("ModelOutput", back_populates="embeddings")


class RequestLog(Base):
    __tablename__ = "request_logs"

    id = Column(Integer, primary_key=True)
    time = Column(DateTime, default=datetime.utcnow)
    resource = Column(String)
    request = Column(JSON)
    response = Column(JSON)


class SurveyDb:
    def __init__(self, db_url="sqlite:///survey.db"):
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)

    def create_tables(self):
        Base.metadata.create_all(self.engine)

    def save_model(self, model):
        with self.Session() as session:
            session.merge(model)
            session.commit()

    def save_prompt(self, prompt):
        with self.Session() as session:
            session.merge(Prompt(**prompt.__dict__))
            session.commit()

    def save_model_output(self, model_output):
        with self.Session() as session:
            if model_output.id:
                db_output = session.query(ModelOutput).get(model_output.id)
                if db_output:
                    for key, value in model_output.__dict__.items():
                        setattr(db_output, key, value)
            else:
                db_output = ModelOutput(**model_output.__dict__)
                session.add(db_output)
            session.commit()
            return db_output.id

    def get_model_output(self, output_id):
        with self.Session() as session:
            return session.query(ModelOutput).get(output_id)

    def models(self):
        with self.Session() as session:
            return session.query(Model).all()

    def get_model(self, model_id):
        with self.Session() as session:
            return session.query(Model).get(model_id)

    def get_prompt(self, prompt_id):
        with self.Session() as session:
            return session.query(Prompt).get(prompt_id)

    def model_outputs(self):
        with self.Session() as session:
            return (
                session.query(ModelOutput)
                .join(Embedding)
                .options(joinedload(ModelOutput.embeddings))
                .all()
            )

    def prompts(self):
        with self.Session() as session:
            return session.query(Prompt).all()

    def log_request(self, resource, request, response):
        with self.Session() as session:
            log = RequestLog(
                resource=resource,
                request=json.dumps(request),
                response=json.dumps(response),
            )
            session.add(log)
            session.commit()
            return log.id

    def save_embedding(self, embedding):
        with self.Session() as session:
            db_embedding = Embedding(**embedding.__dict__)
            session.add(db_embedding)
            session.commit()

    # Additional utility methods
    def get_embeddings_for_output(self, output_id):
        with self.Session() as session:
            return session.query(Embedding).filter_by(output_id=output_id).all()

    def get_request_log(self, log_id):
        with self.Session() as session:
            return session.query(RequestLog).get(log_id)

    def get_model_outputs_by_model(self, model_id):
        with self.Session() as session:
            return session.query(ModelOutput).filter_by(model=model_id).all()


#     # ... other methods would be similarly refactored ...
#
#
# import json
# import re
# import sqlite3
# from collections import defaultdict
# from dataclasses import dataclass
# from decimal import Decimal
# from typing import Dict
#
# import marshmallow
# import numpy as np
#
#
# @dataclass
# class Model:
#     id: str
#     name: str
#     description: str
#     context_length: int
#     pricing: Dict
#
#     @classmethod
#     def from_row(cls, row):
#         return cls(
#             id=row["id"],
#             name=row["name"],
#             description=row["description"],
#             context_length=row["context_length"],
#             pricing=json.loads(row["pricing"]),
#         )
#
#
# @dataclass
# class ModelOutput:
#     id: int
#     model: str
#     content: str
#     embedding: np.array = None
#     usage: Dict = None
#     evaluation: str = None
#     request_id: int = None
#
#     @property
#     def score(self):
#         if not self.evaluation:
#             return None
#
#         if match := re.search(r'"score":\s*([0-9.]+)', self.evaluation):
#             return float(match.group(1))
#
#     @classmethod
#     def from_row(cls, row):
#         return cls(
#             id=row["id"],
#             content=row["content"],
#             embedding=(np.frombuffer(row["embedding"]) if row["embedding"] else None),
#             model=row["model"],
#             usage=json.loads(row["usage"]),
#             evaluation=row["evaluation"],
#             request_id=row["request_id"],
#         )
#
#     @classmethod
#     def from_completion(cls, completion, model, request_id):
#         assert model.id == completion.model
#
#         message = completion.choices[0].message
#         usage = completion.usage
#         pricing = model.pricing
#
#         return cls(
#             id=None,
#             content=message.content,
#             model=completion.model,
#             usage={
#                 "prompt_tokens": usage.prompt_tokens,
#                 "completion_tokens": usage.completion_tokens,
#                 "total_tokens": usage.completion_tokens,
#                 "total_cost": (
#                     usage.prompt_tokens * Decimal(pricing["prompt"])
#                     + usage.completion_tokens * Decimal(pricing["completion"])
#                 ),
#             },
#             request_id=request_id,
#         )
#
#
# @dataclass
# class Embedding:
#     output_id: int
#     embedding: np.array
#     model: str
#     request_id: int
#
#
# @dataclass
# class Prompt:
#     id: str
#     prompt: str = None
#     marking_prompt: str = None
#
#     @classmethod
#     def from_row(cls, row):
#         return cls(
#             id=row["id"], prompt=row["prompt"], marking_prompt=row["marking_prompt"]
#         )
#
#
# class DecimalEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, Decimal):
#             return str(obj)
#         return super().default(obj)
#
#
# class SurveyDb:
#     def __init__(self, filename="survey.db"):
#         self.sqlite = sqlite3.connect(filename)
#
#     def query(self, query, *args):
#         cursor = self.sqlite.execute(query, args)
#         cursor.row_factory = sqlite3.Row
#         return cursor
#
#     def create_tables(self):
#         self.sqlite.execute(
#             """CREATE TABLE IF NOT EXISTS models (
#             id TEXT PRIMARY KEY,
#             name TEXT,
#             description TEXT,
#             context_length INTEGER,
#             pricing TEXT
#         )"""
#         )
#
#         self.sqlite.execute(
#             """CREATE TABLE IF NOT EXISTS prompts (
#             id TEXT PRIMARY KEY,
#             prompt TEXT
#             marking_prompt TEXT
#         )"""
#         )
#
#         self.sqlite.execute(
#             """CREATE TABLE IF NOT EXISTS model_outputs (
#             id INTEGER PRIMARY KEY,
#             content TEXT,
#             model TEXT,
#             usage TEXT,
#             evaluation TEXT
#             request_id INTEGER REFERENCES request_logs(id)
#             )"""
#         )
#
#         self.sqlite.execute(
#             """CREATE TABLE IF NOT EXISTS embeddings (
#                     id INTEGER PRIMARY KEY,
#                     output_id INTEGER REFERENCES model_outputs(id),
#                     embedding BLOB,
#                     model TEXT,
#                     request_id INTEGER REFERENCES request_logs(id),
#                     unique(model, output_id)
#                 )
#                 """
#         )
#
#         self.sqlite.execute(
#             """CREATE TABLE IF NOT EXISTS request_logs (
#                     id INTEGER PRIMARY KEY,
#                     time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#                     resource TEXT,
#                     request BLOB,
#                     response BLOB
#             )"""
#         )
#
#         self.sqlite.commit()
#
#     def save_model(self, model):
#         self.sqlite.execute(
#             """INSERT INTO models VALUES (?, ?, ?, ?, ?)
#             on conflict(id) do update set
#             name = excluded.name,
#             description = excluded.description,
#             context_length = excluded.context_length,
#             pricing = excluded.pricing
#             """,
#             (
#                 model.id,
#                 model.name,
#                 model.description,
#                 model.context_length,
#                 json.dumps(model.pricing),
#             ),
#         )
#         self.sqlite.commit()
#
#     def save_prompt(self, prompt):
#         self.sqlite.execute(
#             """INSERT INTO prompts (id, prompt, marking_prompt) VALUES (?, ?, ?)
#             on conflict(id) do update set
#             prompt = excluded.prompt,
#             marking_prompt = excluded.marking_prompt
#             """,
#             (prompt.id, prompt.prompt, prompt.marking_prompt),
#         )
#         self.sqlite.commit()
#
#     def save_model_output(self, model_output):
#         if model_output.embedding is not None:
#             assert isinstance(model_output.embedding, np.ndarray), type(
#                 model_output.embedding
#             )
#         if model_output.id is not None:
#             self.sqlite.execute(
#                 """UPDATE model_outputs
#                 SET content = ?,
#                 embedding = ?,
#                 model = ?,
#                 usage = ?,
#                 evaluation = ?
#                 WHERE id = ?
#                 """,
#                 (
#                     model_output.content,
#                     model_output.embedding,
#                     model_output.model,
#                     json.dumps(model_output.usage, cls=DecimalEncoder),
#                     model_output.evaluation,
#                     model_output.id,
#                 ),
#             )
#             output_id = model_output.id
#         else:
#             cursor = self.sqlite.execute(
#                 """INSERT INTO model_outputs
#                 (content, embedding, model, usage, evaluation)
#                 VALUES (?, ?, ?, ?, ?)
#                 RETURNING id
#                 """,
#                 (
#                     model_output.content,
#                     model_output.embedding,
#                     model_output.model,
#                     json.dumps(model_output.usage, cls=DecimalEncoder),
#                     model_output.evaluation,
#                 ),
#             )
#             (output_id,) = cursor.fetchone()
#         self.sqlite.commit()
#         return output_id
#
#     def get_model_output(self, output_id):
#         cursor = self.query(
#             """SELECT
#                 model_outputs.id,
#                 model_outputs.content,
#                 embeddings.embedding,
#                 model_outputs.model,
#                 model_outputs.usage,
#                 model_outputs.evaluation,
#                 model_outputs.request_id
#             FROM model_outputs
#             join embeddings on embeddings.output_id = model_outputs.id WHERE id=?""",
#             output_id,
#         )
#         row = cursor.fetchone()
#         if row is None:
#             return None
#
#         return ModelOutput.from_row(row)
#
#     def models(self):
#         for row in self.query("SELECT * FROM models"):
#             yield Model.from_row(row)
#
#     def get_model(self, model_id):
#         cursor = self.query("SELECT * FROM models WHERE id=?", model_id)
#         row = cursor.fetchone()
#         if row is None:
#             return None
#
#         return Model.from_row(row)
#
#     def get_prompt(self, prompt_id):
#         cursor = self.query("SELECT * FROM prompts WHERE id=?", prompt_id)
#         row = cursor.fetchone()
#         if row is None:
#             return None
#
#         return Prompt.from_row(row)
#
#     def model_outputs(self):
#         for row in self.query(
#             """
#                 SELECT
#                     model_outputs.id,
#                     model_outputs.content,
#                     embeddings.embedding,
#                     model_outputs.model,
#                     model_outputs.usage,
#                     model_outputs.evaluation,
#                     model_outputs.request_id
#                 FROM model_outputs
#                 JOIN embeddings ON embeddings.output_id = model_outputs.id
#             """
#         ):
#             yield ModelOutput.from_row(row)
#
#     def prompts(self):
#         for row in self.query("SELECT * FROM prompts"):
#             yield Prompt.from_row(row)
#
#     def log_request(self, resource, request, response):
#         cursor = self.sqlite.execute(
#             """INSERT INTO request_logs (resource, request, response)
#             VALUES (?, ?, ?)
#             RETURNING id
#             """,
#             (resource, json.dumps(request), json.dumps(response)),
#         )
#         (output_id,) = cursor.fetchone()
#         self.sqlite.commit()
#         return output_id
#
#     def save_embedding(self, embedding):
#         self.sqlite.execute(
#             """INSERT INTO embeddings (output_id, embedding, model, request_id)
#             VALUES (?, ?, ?, ?)
#             """,
#             (
#                 embedding.output_id,
#                 embedding.embedding,
#                 embedding.model,
#                 embedding.request_id,
#             ),
#         )
#         self.sqlite.commit()
#
#
import marshmallow


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
        return ModelOutput(id=None, **data)


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
