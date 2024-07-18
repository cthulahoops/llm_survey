from llm_survey.query import log_request_details, reuse_request_if_possible


@reuse_request_if_possible
@log_request_details
def fake_completion(model, prompt):
    return {"model": model, "result": "hello, world"}


def test_reuse_logged_request(in_memory_db):
    request_id_1, result_1 = fake_completion(in_memory_db, "test-model", "Hi")
    request_id_2, result_2 = fake_completion(in_memory_db, "test-model", "Hi")

    assert request_id_1 == request_id_2
    assert result_1 == result_2


def test_different_logged_request(in_memory_db):
    request_id_1, result_1 = fake_completion(in_memory_db, "test-model-1", "Hi")
    request_id_2, result_2 = fake_completion(in_memory_db, "test-model-2", "Hi")

    assert request_id_1 != request_id_2
