import functools
import json
import os
import sqlite3


def log_request_details(f):
    @functools.wraps(f)
    def wrapper(db, *args, **kwargs):
        result = f(*args, **kwargs)
        request_id = db.log_request(__name__, (args, kwargs), result.to_dict())
        return (request_id, result)

    return wrapper


def get_client():
    import openai

    return openai.Client(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )


def sqlite_cache(db_file):
    def cache_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with sqlite3.connect(db_file) as conn:
                c = conn.cursor()
                c.execute(
                    """CREATE TABLE IF NOT EXISTS cache (args TEXT PRIMARY KEY, result TEXT)"""
                )

                jsoned_args = json.dumps(
                    {
                        "args": args,
                        "kwargs": kwargs,
                    }
                )
                c.execute("SELECT result FROM cache WHERE args=?", (jsoned_args,))
                result = c.fetchone()
                if result is not None:
                    return json.loads(result[0])

                result = func(*args, **kwargs)
                c.execute(
                    "INSERT INTO cache VALUES (?, ?)", (jsoned_args, result.to_json())
                )
                conn.commit()
                return result

        return wrapper

    return cache_decorator


@log_request_details
def get_completion(model, prompt):
    return get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )


@sqlite_cache("evaluation_cache.db")
def get_model_response(model, prompt):
    request_id, completion = get_completion(model, prompt)
    return completion.choices[0].message.content


@log_request_details
def get_models():
    return get_client().models.list()


@log_request_details
# @sqlite_cache("embedding_cache.db")
def create_embedding(model, content):
    import openai

    client = openai.Client(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    return client.embeddings.create(
        model=model,
        input=content,
    )


def embed_content(db, content, model="text-embedding-3-small"):
    import numpy as np

    request_id, response = create_embedding(db, model, content)
    return request_id, np.array(response.data[0].embedding)
