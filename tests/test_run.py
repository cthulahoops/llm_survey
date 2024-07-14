import pytest
from click.testing import CliRunner
from llm_survey.data import Model, Prompt
from llm_survey.run import run


def test_run_empty_db(mock_client, mock_db):
    runner = CliRunner()
    result = runner.invoke(run)

    assert result.exit_code == 0
    mock_client.assert_not_called()

    assert len(mock_db.model_outputs()) == 0
    assert len(mock_db.models()) == 0


def test_run_with_model(mock_client, mock_db):
    model = Model(
        id="test_model",
        name="Test Model",
        description="A test model",
        context_length=1024,
        pricing={"prompt": 0.001, "completion": 0.002},
    )
    mock_db.insert(model)

    prompt = Prompt(id="marshmallow", prompt="Test prompt")
    mock_db.insert(prompt)

    runner = CliRunner()
    result = runner.invoke(run, ["--count", "1"], catch_exceptions=False)

    outputs = mock_db.model_outputs()
    assert len(outputs) == 1
    assert outputs[0].model == "test_model"
    assert outputs[0].content == "Response to: Test prompt"
    assert outputs[0].request_id is not None


if __name__ == "__main__":
    pytest.main([__file__])
