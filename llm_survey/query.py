import functools
import json
import os
import sqlite3

import openai

client = openai.Client(
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
                    return result[0]

                result = func(*args, **kwargs)
                c.execute("INSERT INTO cache VALUES (?, ?)", (jsoned_args, str(result)))
                conn.commit()
                return str(result)

        return wrapper

    return cache_decorator


@sqlite_cache("evaluation_cache.db")
def get_model_response(model, prompt):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content


def get_models():
    return client.models.list().data
