from unittest.mock import MagicMock, patch

import pytest
from llm_survey.data import SurveyDb


@pytest.fixture
def in_memory_db():
    db = SurveyDb("sqlite://")
    db.create_tables()
    return db


@pytest.fixture
def mock_db(in_memory_db):
    with patch("llm_survey.data.create_engine") as mock_survey_db:
        mock_survey_db.return_value = in_memory_db.engine
        yield in_memory_db


@pytest.fixture
def mock_client():
    with patch("llm_survey.query.get_client") as mock_get_client:
        client = MagicMock()

        def create_chat_completion(model, messages):
            response = "Response to: " + messages[0]["content"]
            mock_completion = MagicMock()
            mock_completion.model = model
            mock_completion.choices = [MagicMock(message=MagicMock(content=response))]
            mock_completion.error = None
            mock_completion.usage = MagicMock(
                prompt_tokens=2,
                completion_tokens=5,
                total_tokens=7,
            )
            mock_completion.to_dict.return_value = {
                "model": mock_completion.model,
                "choices": [{"message": {"content": "Test completion"}}],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 5,
                    "total_tokens": 7,
                },
            }
            return mock_completion

        client.chat.completions.create = create_chat_completion
        mock_get_client.return_value = client
        yield client
