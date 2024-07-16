import functools
import os


def log_request_details(f):
    @functools.wraps(f)
    def wrapper(db, *args, **kwargs):
        result = f(*args, **kwargs)
        request_id = db.log_request(__name__, (args, kwargs), result.to_dict())
        return (request_id, result)

    return wrapper


def get_openrouter_client():
    import openai

    return openai.Client(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )


def get_openai_client():
    import openai

    return openai.Client(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )


@log_request_details
def get_completion(model, prompt):
    return get_openrouter_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )


@log_request_details
def get_models():
    return get_openrouter_client().models.list()


@log_request_details
# @sqlite_cache("embedding_cache.db")
def create_embedding(model, content):
    return get_openai_client().embeddings.create(
        model=model,
        input=content,
    )


def embed_content(db, content, model="text-embedding-3-small"):
    import numpy as np

    request_id, response = create_embedding(db, model, content)
    return request_id, np.array(response.data[0].embedding)
