import json
import numpy as np
import google.generativeai as genai
from model_config import configure_genai

def load_help_data(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

def generate_embeddings(data):
    embeddings = []
    for entry in data:
        res = genai.embed_content(
            model="models/embedding-001",
            content=entry["content"],
            task_type="retrieval_document"
        )
        entry["embedding"] = res["embedding"]
        embeddings.append(res["embedding"])
    return data, np.array(embeddings)

def main():
    configure_genai()
    data = load_help_data("data/google_help_data.json")
    data, embedding_array = generate_embeddings(data)

    # Save embeddings (optional - can use pickle or npy)
    with open("data/embedded_data.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Embedded {len(data)} entries.")

if __name__ == "__main__":
    main()
