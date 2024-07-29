import numpy as np
import pytest
from llm_survey.data import Embedding, Evaluation, ModelOutput, SurveyDb


@pytest.fixture
def db():
    db = SurveyDb("sqlite://")
    db.create_tables()
    yield db


@pytest.fixture
def model_output(db):
    output = ModelOutput(id=None, model="test", content="Hello, World")
    with db.Session() as session:
        session.add(output)
        session.commit()
        session.refresh(output)
    return output


@pytest.mark.parametrize(
    "output", [ModelOutput(id=None, model="test", content="Hello, World")]
)
def test_save_and_retrieve_output(db, output):
    content = output.content

    db.insert(output)

    outputs = db.model_outputs()

    assert len(outputs) == 1
    actual_output = outputs[0]

    assert content == actual_output.content
    assert actual_output.id == 1


def test_add_evaluation(db, model_output):
    completion = "Great greeting. 10/10."
    evaluation = Evaluation(
        content=completion,
        model_output_id=model_output.id,
        model="just-testing",
    )
    db.insert(evaluation)

    model_output = db.get_model_output(model_output.id)

    assert model_output.evaluation(model="just-testing").content == completion


def test_insert_embedding(db):
    embedding = np.array(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float64
    )

    output = ModelOutput(id=None, model="test", content="Hello, World")
    db.insert(output)
    [output] = db.model_outputs()

    embedding_obj = Embedding(
        output_id=output.id,
        model="fake_and_random",
        embedding=embedding,
    )
    db.insert(embedding_obj)

    [output] = db.model_outputs()
    assert all(output.embedding == embedding)


def test_log_and_retrieve_request(in_memory_db):
    in_memory_db.log_request(
        "get_completion", (("something-gpt", "prompt"), {"extra": 8}), "c"
    )
    response = in_memory_db.get_logged_request(
        "get_completion", [["something-gpt", "prompt"], {"extra": 8}]
    )
    assert response.response == "c"
