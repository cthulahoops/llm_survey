import json
import re
from collections import defaultdict
from datetime import datetime
from decimal import Decimal

import numpy as np
import sqlalchemy
from sqlalchemy import BLOB, JSON, Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import declarative_base, joinedload, relationship, sessionmaker

Base = declarative_base()


class Model(Base):
    __tablename__ = "models"

    id = Column(String, primary_key=True)
    name = Column(String)
    description = Column(String)
    context_length = Column(Integer)
    pricing = Column(JSON)

    @classmethod
    def from_openai(cls, openai_model):
        return Model(
            id=openai_model["id"],
            name=openai_model["name"],
            description=openai_model["description"],
            context_length=openai_model["context_length"],
            pricing=openai_model["pricing"],
        )


class Prompt(Base):
    __tablename__ = "prompts"

    id = Column(String, primary_key=True)
    prompt = Column(String)
    evaluation_model = Column(String)
    marking_scheme = Column(String)
    model_outputs = relationship("ModelOutput", back_populates="prompt")

    def to_dict(self):
        return {
            "id": self.id,
            "prompt": self.prompt,
            "evaluation_model": self.evaluation_model,
            "marking_scheme": self.marking_scheme,
        }


class ModelOutput(Base):
    __tablename__ = "model_outputs"

    id = Column(Integer, primary_key=True)
    content = Column(String)
    model = Column(String)
    usage = Column(JSON)
    request_id = Column(Integer, ForeignKey("request_logs.id"))
    prompt_id = Column(String, ForeignKey("prompts.id"))

    prompt = relationship("Prompt", back_populates="model_outputs")
    embeddings = relationship("Embedding", back_populates="model_output")
    evaluations = relationship("Evaluation", back_populates="model_output")

    @property
    def embedding(self):
        if self.embeddings:
            return np.frombuffer(self.embeddings[0].embedding)
        return None

    @property
    def evaluation(self):
        if self.evaluations:
            return self.evaluations[0].content
        return None

    @property
    def score(self):
        if not self.evaluation:
            return None

        if match := re.search(r'"score":\s*([0-9.]+)', self.evaluation):
            return float(match.group(1))

    @classmethod
    def from_completion(cls, completion, model, request_id):
        assert model.id == completion["model"]

        message = completion["choices"][0]["message"]
        usage = completion["usage"]
        pricing = model.pricing

        prompt_tokens = usage["prompt_tokens"]
        completion_tokens = usage["completion_tokens"]

        return cls(
            id=None,
            content=message["content"],
            model=completion["model"],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "total_cost": str(
                    prompt_tokens * Decimal(pricing["prompt"])
                    + completion_tokens * Decimal(pricing["completion"])
                ),
            },
            request_id=request_id,
        )


class Embedding(Base):
    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True)
    output_id = Column(Integer, ForeignKey("model_outputs.id"))
    embedding = Column(BLOB)
    model = Column(String)
    request_id = Column(Integer, ForeignKey("request_logs.id"))

    model_output = relationship("ModelOutput", back_populates="embeddings")


class Evaluation(Base):
    __tablename__ = "evaluations"

    id = Column(Integer, primary_key=True)
    model_output_id = Column(Integer, ForeignKey("model_outputs.id"))
    content = Column(String)
    model = Column(String)
    usage = Column(JSON)
    request_id = Column(Integer, ForeignKey("request_logs.id"))

    model_output = relationship("ModelOutput", back_populates="evaluations")

    @classmethod
    def from_completion(cls, model_output, evaluation_model, completion, request_id):
        assert evaluation_model.id == completion["model"]

        message = completion["choices"][0]["message"]
        usage = completion["usage"]
        pricing = evaluation_model.pricing

        prompt_tokens = usage["prompt_tokens"]
        completion_tokens = usage["completion_tokens"]

        return cls(
            id=None,
            model_output_id=model_output.id,
            content=message["content"],
            model=completion["model"],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "total_cost": str(
                    prompt_tokens * Decimal(pricing["prompt"])
                    + completion_tokens * Decimal(pricing["completion"])
                ),
            },
            request_id=request_id,
        )


class RequestLog(Base):
    __tablename__ = "request_logs"

    id = Column(Integer, primary_key=True)
    time = Column(DateTime, default=datetime.utcnow)
    resource = Column(String)
    request = Column(JSON)
    response = Column(JSON)


class SurveyDb:
    def __init__(self, db_url="sqlite:///survey.db"):
        self.engine = sqlalchemy.create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)

    def create_tables(self):
        Base.metadata.create_all(self.engine)

    def insert(self, obj):
        with self.Session() as session:
            session.add(obj)
            session.commit()

    def save_prompt(self, prompt):
        with self.Session() as session:
            old_prompt = session.get(Prompt, prompt["id"])
            if old_prompt is None:
                session.add(Prompt(**prompt))
            else:
                for key, value in prompt.items():
                    setattr(old_prompt, key, value)
            session.commit()

    def delete_prompt(self, prompt_id):
        with self.Session() as session:
            prompt = session.get(Prompt, prompt_id)
            session.delete(prompt)
            session.commit()

    def get_model_output(self, output_id):
        with self.Session() as session:
            return session.get(
                ModelOutput,
                output_id,
                options=[
                    joinedload(ModelOutput.embeddings),
                    joinedload(ModelOutput.evaluations),
                ],
            )

    def models(self):
        with self.Session() as session:
            return session.query(Model).all()

    def get_model(self, model_id):
        with self.Session() as session:
            return session.get(Model, model_id)

    def get_prompt(self, prompt_id):
        with self.Session() as session:
            return session.get(Prompt, prompt_id)

    def get_prompt_outputs(self, prompt_id):
        with self.Session() as session:
            return session.get(
                Prompt,
                prompt_id,
                options=[joinedload(Prompt.model_outputs)],
            )

    def model_outputs(self):
        with self.Session() as session:
            return (
                session.query(ModelOutput)
                .options(
                    [
                        joinedload(ModelOutput.embeddings),
                        joinedload(ModelOutput.evaluations),
                    ]
                )
                .all()
            )

    def prompts(self):
        with self.Session() as session:
            return session.query(Prompt).all()

    def log_request(self, resource, request, response):
        with self.Session() as session:
            log = RequestLog(
                resource=resource,
                request=request,
                response=response,
            )
            session.add(log)
            session.commit()
            return log.id

    def get_logged_request(self, resource, request):
        with self.Session() as session:
            return (
                session.query(RequestLog)
                .filter_by(resource=resource, request=json.dumps(request))
                .first()
            )


def groupby(data, key):
    result = defaultdict(list)
    for item in data:
        result[key(item)].append(item)
    return result
