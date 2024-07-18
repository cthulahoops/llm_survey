from click.testing import CliRunner
from llm_survey.models import models


def test_fetch_models(mock_client, mock_db):
    runner = CliRunner()
    runner.invoke(models, catch_exceptions=False)

    models_in_db = mock_db.models()

    assert len(models_in_db) > 0
