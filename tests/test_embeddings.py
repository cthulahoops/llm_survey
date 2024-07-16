import numpy as np
from click.testing import CliRunner
from llm_survey.data import ModelOutput
from llm_survey.embeddings import embeddings


def test_run_one_embedding(mock_client, mock_db):
    output = ModelOutput(model="test-model", content="Evaluate this")
    mock_db.insert(output)

    runner = CliRunner()
    runner.invoke(embeddings, catch_exceptions=False)

    mock_client.return_value.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small", input="Evaluate this"
    )

    [model_output] = mock_db.model_outputs()

    assert all(model_output.embedding == np.array([0.2, 0.3]))
