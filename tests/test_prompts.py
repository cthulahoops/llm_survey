import io
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


@pytest.mark.parametrize(
    "format_flag,loader",
    [
        ("--format json", json.loads),
        ("--format yaml", lambda data: YAML().load(data)),
    ],
)
def test_export_prompt(format_flag, loader):
    runner = CliRunner()
    result = runner.invoke(
        prompts, ["export", "marshmallow"] + format_flag.split(), catch_exceptions=False
    )
    assert result.exit_code == 0

    output = loader(result.output)

    assert output["id"] == "marshmallow"
    assert output["prompt"] == "How many marshmallows are there?"
    assert output["evaluation_model"] == "mega-good-model"
    assert output["marking_scheme"] == "Award one mark for every marshmallow"


def dump_yaml(data):
    yaml = YAML()
    buffer = io.StringIO()
    yaml.dump(data, buffer)
    return buffer.getvalue()


@pytest.mark.parametrize(
    "format_flag,dumper",
    [
        ("--format json", json.dumps),
        ("--format yaml", dump_yaml),
    ],
)
def test_import_prompt(mock_db, format_flag, dumper):
    prompt = {
        "id": "new-prompt",
        "prompt": "What is the answer to the ultimate question?",
        "evaluation_model": "super-duper-model",
        "marking_scheme": "Award one mark for the answer",
    }
    runner = CliRunner()
    result = runner.invoke(
        prompts,
        ["import"] + format_flag.split(),
        input=dumper(prompt),
        catch_exceptions=False,
    )
    assert result.exit_code == 0

    prompt = mock_db.get_prompt("new-prompt")
    assert prompt.id == "new-prompt"


@pytest.mark.parametrize(
    "format_flag,dumper",
    [
        ("--format json", json.dumps),
        ("--format yaml", dump_yaml),
    ],
)
def test_import_update(mock_db, format_flag, dumper):
    prompt = {
        "id": "marshmallow",
        "prompt": "What is the answer to the ultimate question?",
        "evaluation_model": "super-duper-model",
        "marking_scheme": "Award one mark for the answer",
    }
    runner = CliRunner()
    result = runner.invoke(
        prompts,
        ["import"] + format_flag.split(),
        input=dumper(prompt),
        catch_exceptions=False,
    )
    assert result.exit_code == 0

    prompt = mock_db.get_prompt("marshmallow")
    assert prompt.id == "marshmallow"


def test_list_prompts():
    runner = CliRunner()
    result = runner.invoke(prompts, ["list"], catch_exceptions=False)
    assert result.exit_code == 0

    assert "marshmallow" in result.output
