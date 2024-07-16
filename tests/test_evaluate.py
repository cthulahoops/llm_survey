import pytest
from click.testing import CliRunner
from llm_survey.data import Model, ModelOutput
from llm_survey.evaluate import DEFAULT_EVALUATION_MODEL, evaluate


@pytest.fixture(autouse=True)
def evaluation_model(mock_db):
    model = Model(
        id=DEFAULT_EVALUATION_MODEL,
        pricing={
            "prompt": "0.01",
            "completion": "1.00",
        },
    )
    mock_db.insert(model)


def test_evaluate_empty_db(mock_client, mock_db):
    runner = CliRunner()
    runner.invoke(evaluate, catch_exceptions=False)

    mock_client.assert_not_called()


def test_evaluate_one(mock_client, mock_db):
    runner = CliRunner()

    output = ModelOutput(model="test-model", content="Evaluate this")
    mock_db.insert(output)

    runner.invoke(evaluate, catch_exceptions=False)

    mock_client.return_value.chat.completions.create.assert_called_once()

    assert len(mock_db.model_outputs()) == 1
