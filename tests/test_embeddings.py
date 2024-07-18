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


def test_run_two_identical_embeddings(mock_client, mock_db):
    for _ in range(2):
        output = ModelOutput(model="test-model", content="Evaluate this")
        mock_db.insert(output)

    runner = CliRunner()
    runner.invoke(embeddings, catch_exceptions=False)

    mock_client.return_value.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small", input="Evaluate this"
    )

    [output1, output2] = mock_db.model_outputs()

    assert output1.embeddings[0].request_id == output2.embeddings[0].request_id
    assert all(output1.embedding == np.array([0.2, 0.3]))
    assert all(output2.embedding == np.array([0.2, 0.3]))
