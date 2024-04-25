import functools
import json
import os
import sqlite3

import openai

from llm_survey.data import load_data, save_data


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


client = openai.Client(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)


@sqlite_cache("evaluation_cache.db")
def get_evaluation(model, prompt, response_to_evaluate):
    print("Calling out to :", model)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt + response_to_evaluate},
        ],
    )
    try:
        return completion.choices[0].message.content
    except Exception:
        print(completion)
        raise


def main():
    data = load_data("embeddings.jsonl")

    prompt = open("evaluation.md").read()

    data = list(data)

    count = 0
    for item in data:
        print("#", count, " - ", item.model)

        content = get_evaluation("openai/gpt-4-turbo", prompt, item.content)
        item.evaluation = content

        count += 1
        if count % 3 == 0:
            save_data("evaluation.jsonl", data)

    save_data("evaluation.jsonl", data)


if __name__ == "__main__":
    main()
