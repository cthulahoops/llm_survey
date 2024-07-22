from click.testing import CliRunner
from llm_survey.cli import cli


def test_fetch_models(mock_client, mock_db):
    runner = CliRunner()
    runner.invoke(cli, ["models", "fetch"], catch_exceptions=False)

    models_in_db = mock_db.models()

    assert len(models_in_db) > 0

    model_ids = {model.id for model in models_in_db}

    assert "openrouter/auto" not in model_ids
