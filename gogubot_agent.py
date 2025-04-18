
import json
import numpy as np
import faiss
import google.generativeai as genai

class GogubotAgent:
    def __init__(self, data_path, api_key):
        self.api_key = api_key
        self.data_path = data_path
        self.data = self._load_data()
        self.embeddings = [entry["embedding"] for entry in self.data]
        self.index = self._build_index()
        self.history = []
        self._configure_gemini()

    def _configure_gemini(self):
        genai.configure(api_key=self.api_key)

    def _load_data(self):
        with open(self.data_path, "r") as f:
            return json.load(f)

    def _build_index(self):
        dim = len(self.embeddings[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(self.embeddings))
        return index

    def ask(self, query, k=1):
        self.history.append({"user": query})
        # Embed user query
        res = genai.embed_content(
            model="models/embedding-001",
            content=query,
            task_type="retrieval_query"
        )
        query_vector = np.array([res["embedding"]])
        D, I = self.index.search(query_vector, k)
        matched = self.data[I[0][0]]

        # Use Gemini to generate human-friendly response
        prompt = f"""
You are a helpful assistant. A user has a question:
"{query}"

Here is a helpful Google Help snippet you found:
"{matched['content']}"

Answer the user in a clear, friendly, and helpful way using the snippet above.
"""
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        response = gemini_model.generate_content(prompt).text
        self.history.append({"bot": response})
        return response

    def chat_history(self):
        return self.history
