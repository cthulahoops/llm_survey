from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner
from llm_survey.cli import cli
from llm_survey.data import Model, Prompt
from llm_survey.evaluate import DEFAULT_EVALUATION_MODEL


@pytest.fixture(autouse=True)
def build_db(mock_db):
    prompt = Prompt(
        id="test-prompt",
        prompt="Does this test work?",
        marking_scheme="It should pass",
    )
    mock_db.insert(prompt)
    model = Model(
        id=DEFAULT_EVALUATION_MODEL,
        pricing={
            "prompt": "0.01",
            "completion": "1.00",
        },
    )
    mock_db.insert(model)


@pytest.fixture
def mock_open():
    with patch("__builtins__.open") as os_open:
        os_open.return_value = Mock()


def test_build(mock_client, mock_db):
    runner = CliRunner()
    runner.invoke(cli, ["build", "test-prompt"], catch_exceptions=False)
