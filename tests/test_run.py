from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from llm_survey.data import Model, Prompt, SurveyDb
from llm_survey.run import run


@pytest.fixture
def in_memory_db():
    db = SurveyDb("sqlite://")
    db.create_tables()
    return db


@patch("llm_survey.run.SurveyDb")
@patch("llm_survey.run.get_completion")
def test_run_empty_db(mock_get_completion, mock_survey_db, in_memory_db):
    mock_survey_db.return_value = in_memory_db

    runner = CliRunner()
    result = runner.invoke(run)

    assert result.exit_code == 0
    mock_get_completion.assert_not_called()

    assert len(in_memory_db.model_outputs()) == 0
    assert len(in_memory_db.models()) == 0


@patch("llm_survey.run.SurveyDb")
@patch("llm_survey.run.get_completion")
def test_run_with_model(mock_get_completion, mock_survey_db, in_memory_db):
    mock_survey_db.return_value = in_memory_db

    model = Model(
        id="test_model",
        name="Test Model",
        description="A test model",
        context_length=1024,
        pricing={"prompt": 0.001, "completion": 0.002},
    )
    in_memory_db.insert(model)

    prompt = Prompt(id="marshmallow", prompt="Test prompt")
    in_memory_db.insert(prompt)

    mock_completion = MagicMock()
    mock_completion.model = "test_model"
    mock_completion.choices = [MagicMock(message=MagicMock(content="Test completion"))]
    mock_completion.error = None
    mock_completion.usage = MagicMock(
        prompt_tokens=2,
        completion_tokens=5,
        total_tokens=7,
    )
    mock_get_completion.return_value = (None, mock_completion)

    runner = CliRunner()
    result = runner.invoke(run, ["--count", "1"], catch_exceptions=False)

    mock_get_completion.assert_called_once_with(
        in_memory_db, "test_model", "Test prompt"
    )

    outputs = in_memory_db.model_outputs()
    assert len(outputs) == 1
    assert outputs[0].model == "test_model"
    assert outputs[0].content == "Test completion"


if __name__ == "__main__":
    pytest.main([__file__])
