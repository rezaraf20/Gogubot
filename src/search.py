import json
import numpy as np
import faiss
import google.generativeai as genai
from model_config import configure_genai

def load_embedded_data(path="data/embedded_data.json"):
    with open(path, "r") as f:
        data = json.load(f)
    embeddings = [entry["embedding"] for entry in data]
    return data, np.array(embeddings)

def build_faiss_index(embedding_array):
    dim = embedding_array.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embedding_array)
    return index

def answer_question(query, data, index, k=1):
    res = genai.embed_content(
        model="models/embedding-001",
        content=query,
        task_type="retrieval_query"
    )
    query_vector = np.array([res["embedding"]])
    D, I = index.search(query_vector, k)
    return [data[i] for i in I[0]]

def main():
    configure_genai()
    data, embedding_array = load_embedded_data()
    index = build_faiss_index(embedding_array)

    question = input("❓ Ask something about Google services: ")
    answers = answer_question(question, data, index, k=1)
    for ans in answers:
        print(f"\n✅ {ans['title']} ({ans['service']})\n{ans['content']}")

if __name__ == "__main__":
    main()
