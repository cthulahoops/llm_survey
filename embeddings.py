import argparse
import json
from pathlib import Path

import openai


def read_model_data():
    with open("llm_log.jsonl") as f:
        for line in f:
            data = json.loads(line)

            if "messages" not in data:
                continue

            last_message = data["messages"][-1]

            if last_message["role"] != "assistant":
                continue

            yield {"model": data["model"], "content": last_message["content"]}


client = openai.Client()


def embed_content(content, model="text-embedding-3-small"):
    response = client.embeddings.create(
        model=model,
        input=content,
    )
    return response.data[0].embedding


# with open("embeddings.jsonl", "w") as f:
#     for sample in read_model_data():
#         response = openai.Embedding.create(
#             model="text-embedding-3-small",
#             input=sample["content"],
#         )
#         embedding = response.data[0].embedding
#
#         sample["embedding"] = embedding
#
#         f.write(json.dumps(sample) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="text-embedding-3-small")
    parser.add_argument("output", type=Path)
    parser.add_argument("input", type=Path, nargs="+")
    args = parser.parse_args()

    # If input is markdown:
    results = []

    for input_file in args.input:
        if input_file.suffix == ".md":
            with input_file.open() as f:
                content = f.read()
            embedding = embed_content(content, model=args.model)
            results.append(
                {
                    "content": content,
                    "model": "human",
                    "embedding": embedding,
                }
            )

    with args.output.open("w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()

# model="text-ada-001",
# model="text-babbage-001",
# model="text-curie-001",
# model="text-davinci-002",
