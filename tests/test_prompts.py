import json

import pytest
from click.testing import CliRunner
from llm_survey.data import Prompt
from llm_survey.prompts import prompts


@pytest.fixture(autouse=True)
def prompt_db(mock_db):
    prompt = Prompt(
        id="marshmallow",
        prompt="How many marshmallows are there?",
        marking_scheme="Award one mark for every marshmallow",
    )
    mock_db.insert(prompt)


def test_prompts():
    runner = CliRunner()
    result = runner.invoke(prompts, ["export", "marshmallow"], catch_exceptions=False)
    assert result.exit_code == 0

    output = json.loads(result.output)

    assert output["id"] == "marshmallow"
    assert output["prompt"] == "How many marshmallows are there?"
    assert output["marking_scheme"] == "Award one mark for every marshmallow"
