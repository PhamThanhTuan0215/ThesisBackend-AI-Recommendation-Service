import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

models = genai.list_models()

print("📦 Các model khả dụng với key hiện tại:")
for m in models:
    print("-", m.name)
