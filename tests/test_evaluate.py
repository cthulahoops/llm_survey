import pytest
from llm_survey.data import Model, ModelOutput, Prompt

from conftest import invoke


@pytest.fixture(autouse=True)
def evaluation_db(mock_db):
    prompt = Prompt(
        id="marshmallow",
        prompt="How many marshmallows are there?",
        marking_scheme="Award one mark for every marshmallow",
        evaluation_model="test-evaluator",
    )
    mock_db.insert(prompt)
    model = Model(
        id="test-evaluator",
        pricing={
            "prompt": "0.01",
            "completion": "1.00",
        },
    )
    mock_db.insert(model)


def test_evaluate_empty_db(mock_client, mock_db):
    invoke("evaluate", "marshmallow")

    mock_client.assert_not_called()


def test_evaluate_one(mock_client, mock_db):
    output = ModelOutput(
        prompt_id="marshmallow",
        model="test-model",
        content="Evaluate this",
    )
    mock_db.insert(output)

    invoke("evaluate", "marshmallow")

    mock_client.return_value.chat.completions.create.assert_called_once()

    mock_db.model_outputs()[0].embedding is not None


def test_evaluate_two_identical(mock_client, mock_db):
    for i in range(2):
        output = ModelOutput(
            prompt_id="marshmallow",
            model="test-model",
            content="Evaluate this",
        )
        mock_db.insert(output)

    invoke("evaluate", "marshmallow")
    mock_client.return_value.chat.completions.create.assert_called_once()

    output1, output2 = mock_db.model_outputs()
    output1.evaluations[0].request_id == output2.evaluations[0].request_id
    output1.evaluation == output2.evaluation
