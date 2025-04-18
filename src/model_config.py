import os
import google.generativeai as genai

def configure_genai():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set in environment variables.")
    genai.configure(api_key=api_key)
