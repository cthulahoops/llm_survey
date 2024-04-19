# import os
import openai
import json

# openai.api_key = os.getenv("OPENAI_API_KEY")


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


with open("embeddings.jsonl", "w") as f:
    for sample in read_model_data():
        response = openai.Embedding.create(
            model="text-embedding-3-small",
            input=sample["content"],
        )
        embedding = response.data[0].embedding

        sample["embedding"] = embedding

        f.write(json.dumps(sample) + "\n")


# model="text-ada-001",
# model="text-babbage-001",
# model="text-curie-001",
# model="text-davinci-002",
