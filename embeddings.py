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
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    args = parser.parse_args()

    # If input is markdown:
    if args.input.suffix == ".md":
        with open(args.input) as f:
            content = f.read()
            embedding = embed_content(content, model=args.model)
            output = json.dumps(
                {
                    "content": content,
                    "model": "human",
                    "embedding": embedding,
                }
            )

            if args.output:
                with open(args.output, "w") as f:
                    f.write(output)
            else:
                print(output)


if __name__ == "__main__":
    main()

# model="text-ada-001",
# model="text-babbage-001",
# model="text-curie-001",
# model="text-davinci-002",
