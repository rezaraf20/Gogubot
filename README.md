# Gogubot
The Google Services Assistant

Gogubot is an intelligent chatbot powered by **Gemini** and **FAISS** that helps users navigate and understand Google services like Gmail, Docs, Calendar, Drive, and more.

---

## Features

- ✅ Uses **Gemini Embeddings** for semantic understanding
- ✅ Uses **FAISS** for fast vector search across help documents
- ✅ Generates clear, natural responses with **Gemini Flash**
- ✅ Supports conversational **agent** with memory
- ✅ Modular design for extension (multi-language, APIs, UI)

---

## Project Structure

```
gogubot/
├── data/
│   └── google_help_data.json
├── src/
│   ├── embed.py
│   ├── search.py
│   ├── model_config.py
│   └── gogubot_agent.py
├── notebooks/
│   └── notebookgogubot_enhanced.ipynb
├── requirements.txt
└── README.md
```

