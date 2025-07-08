import requests
import os
import json
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-2.0-flash"

url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
headers = {
    "Content-Type": "application/json",
    "X-goog-api-key": API_KEY
}

user_input = "Tôi đang bị ngứa da và đau bụng"
prompt = f"""
Phân tích câu sau và trích xuất danh sách các triệu chứng có trong đó.

Chỉ trả về kết quả dưới dạng mảng JSON thuần túy gồm các chuỗi, ví dụ:
["triệu chứng 1", "triệu chứng 2"]

Không cần giải thích gì thêm. Không in dấu sao (*), không xuống dòng.

Câu: "{user_input}"
"""

payload = {
    "contents": [
        {
            "parts": [
                {"text": prompt}
            ]
        }
    ]
}

response = requests.post(url, headers=headers, json=payload)

if response.status_code == 200:
    text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    print("📄 Raw text:", text)
    try:
        symptoms = json.loads(text)
        print("✅ Danh sách triệu chứng:", symptoms)
    except Exception as e:
        print("❌ Không phân tích được JSON:", e)
else:
    print("❌ Lỗi:", response.status_code, response.text)
