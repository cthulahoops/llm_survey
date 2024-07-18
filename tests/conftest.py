from unittest.mock import MagicMock, Mock, patch

import pytest
from llm_survey.data import SurveyDb


@pytest.fixture
def in_memory_db():
    db = SurveyDb("sqlite://")
    db.create_tables()
    return db


@pytest.fixture
def mock_db(in_memory_db):
    with patch("sqlalchemy.create_engine") as mock_survey_db:
        mock_survey_db.return_value = in_memory_db.engine
        yield in_memory_db


@pytest.fixture
def mock_client():
    with patch("openai.Client") as mock_get_client:

        def create_chat_completion(model, messages):
            response = "Response to: " + messages[0]["content"]
            mock_completion = Mock()
            mock_completion.to_dict.return_value = {
                "model": model,
                "choices": [{"message": {"content": response}}],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 5,
                    "total_tokens": 7,
                },
            }
            return mock_completion

        def create_embedding(model, input):
            mock_embedding = Mock()
            mock_embedding.to_dict.return_value = {}
            mock_embedding.data = [Mock(embedding=[0.2, 0.3])]
            mock_embedding.to_dict.return_value = {
                "data": [
                    {
                        "embedding": [0.2, 0.3],
                    }
                ]
            }

            return mock_embedding

        openai_client = MagicMock()
        openai_client.chat.completions.create = Mock(side_effect=create_chat_completion)
        openai_client.embeddings.create = Mock(side_effect=create_embedding)
        mock_get_client.return_value = openai_client
        yield mock_get_client
