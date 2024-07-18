from llm_survey.query import log_request_details, reuse_request_if_possible


def test_logged_request(in_memory_db):
    @reuse_request_if_possible
    @log_request_details
    def get_completion(model, prompt):
        return {"model": model, "result": "hello, world"}

    request_id_1, result_1 = get_completion(in_memory_db, "test-model", "Hi")

    request_id_2, result_2 = get_completion(in_memory_db, "test-model", "Hi")

    assert request_id_1 == request_id_2
