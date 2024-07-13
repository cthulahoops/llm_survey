import json
import re
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

    @property
    def score(self):
        if not self.evaluation:
            return None

        if match := re.search(r'"score":\s*([0-9.]+)', self.evaluation):
            return float(match.group(1))


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


def groupby(data, key):
    result = defaultdict(list)
    for item in data:
        result[key(item)].append(item)
    return result
