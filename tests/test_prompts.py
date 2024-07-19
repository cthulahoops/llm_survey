import json

import pytest
from click.testing import CliRunner
from llm_survey.data import Prompt
from llm_survey.prompts import prompts
from ruamel.yaml import YAML


@pytest.fixture(autouse=True)
def prompt_db(mock_db):
    prompt = Prompt(
        id="marshmallow",
        prompt="How many marshmallows are there?",
        marking_scheme="Award one mark for every marshmallow",
        evaluation_model="mega-good-model",
    )
    mock_db.insert(prompt)


def test_export_prompt():
    runner = CliRunner()
    result = runner.invoke(prompts, ["export", "marshmallow"], catch_exceptions=False)
    assert result.exit_code == 0

    output = json.loads(result.output)

    assert output["id"] == "marshmallow"
    assert output["prompt"] == "How many marshmallows are there?"
    assert output["evaluation_model"] == "mega-good-model"
    assert output["marking_scheme"] == "Award one mark for every marshmallow"


def test_export_format_yaml():
    runner = CliRunner()
    result = runner.invoke(
        prompts, ["export", "marshmallow", "--format", "yaml"], catch_exceptions=False
    )
    assert result.exit_code == 0

    yaml = YAML()
    output = yaml.load(result.output)
    assert output["id"] == "marshmallow"
    assert output["prompt"] == "How many marshmallows are there?"
    assert output["evaluation_model"] == "mega-good-model"
    assert output["marking_scheme"] == "Award one mark for every marshmallow"


def test_import_prompt(mock_db):
    prompt = {
        "id": "new-prompt",
        "prompt": "What is the answer to the ultimate question?",
        "evaluation_model": "super-duper-model",
        "marking_scheme": "Award one mark for the answer",
    }
    runner = CliRunner()
    result = runner.invoke(
        prompts,
        ["import"],
        input=json.dumps(prompt),
        catch_exceptions=False,
    )
    assert result.exit_code == 0

    prompt = mock_db.get_prompt("new-prompt")
    assert prompt.id == "new-prompt"


def test_list_prompts():
    runner = CliRunner()
    result = runner.invoke(prompts, ["list"], catch_exceptions=False)
    assert result.exit_code == 0

    assert "marshmallow" in result.output
