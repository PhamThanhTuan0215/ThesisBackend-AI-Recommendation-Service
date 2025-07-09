print("✅ App đang khởi động...")
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import List, Optional

from suggestions import get_all_data_suggestions
import os
import json
import requests
import time
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Gợi ý thuốc theo triệu chứng")

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# uvicorn index:app --reload --port 8088

class SymptomInput(BaseModel):
    user_input: Optional[str] = None
    min_score: Optional[float] = 0.6
    top_k: Optional[int] = 3

@app.get("/")
def root():
    result = {"code": 0, "message": "Welcome to the Disease Prediction API."}
    return JSONResponse(content=result, status_code=200)

@app.post("/suggestions")
async def predict_disease(input: SymptomInput):
    
    print(f"🔍 Nhận được yêu cầu với đầu vào: {input}")
    
    if(not input.user_input or not input.user_input.strip()):
        result = {
            "code": 1,
            "message": "Vui lòng nhập triệu chứng.",
            "data": []
        }
        return JSONResponse(content=result, status_code=400)
    
    # xử lý đầu vào
    symptoms_input, message_error = await run_in_threadpool(
        get_symptoms_input_from_user_input, input.user_input.strip()
    )
    
    if message_error:
        result = {
            "code": 1,
            "message": message_error,
            "data": []
        }
        return JSONResponse(content=result, status_code=400)
    
    if not symptoms_input or len(symptoms_input) == 0:
        result = {
            "code": 1,
            "message": "Không có triệu chứng nào được nhắc đến.",
            "data": []
        }
        return JSONResponse(content=result, status_code=400)
    
    print(f"🔍 Đang xử lý {len(symptoms_input)} triệu chứng: {symptoms_input}")

    data = await run_in_threadpool(get_all_data_suggestions, symptoms_input, input.min_score, input.top_k)


    if not data:
        result = {
            "code": 1,
            "message": "Không tìm thấy dữ liệu gợi ý.",
            "data": []
        }
        return JSONResponse(content=result, status_code=404)
    
    result = {
            "code": 0,
            "message": "Dự đoán bệnh thành công.",
            "data": data
    }
    return JSONResponse(content=result, status_code=200)

from typing import Tuple

def get_symptoms_input_from_user_input(user_input: str) -> Tuple[List[str], str]:
    
    API_KEY = os.getenv("GEMINI_API_KEY")
    MODELS = ["gemini-2.0-flash", "gemini-1.5-flash"]  # fallback nếu model đầu bị lỗi
    MAX_RETRIES = 3 # thử gọi tối đa 3 lần nếu có lỗi
    RETRY_DELAY = 5 # thời gian chờ giữa các lần thử
    if not API_KEY:
        print("Không tìm thấy API key cho Gemini. Vui lòng kiểm tra biến môi trường.")
        return [], "Không tìm thấy API key cho Gemini. Vui lòng kiểm tra biến môi trường."
    if not user_input or not user_input.strip():
        print("Không có đầu vào người dùng.")
        return [], "Không có dữ liệu đầu vào người dùng."
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": API_KEY
    }

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

    for model in MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        for attempt in range(MAX_RETRIES):
            try:
                print(f"Đang gọi model: {model} (attempt {attempt + 1})")
                response = requests.post(url, headers=headers, json=payload)

                if response.status_code == 200:
                    text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
                    print("Phản hồi Gemini:", text)
                    symptoms = json.loads(text)
                    if isinstance(symptoms, list):
                        return symptoms, None
                elif response.status_code == 429:
                    print(f"Bị giới hạn, chờ {RETRY_DELAY}s rồi thử lại...")
                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    print(f"Lỗi không mong muốn: {response.status_code} - {response.text}")
                    break  # nếu không phải lỗi quota thì không thử lại
            except Exception as e:
                print(f"Lỗi khi gọi Gemini: {e}")
                break  # lỗi không mong đợi thì thoát luôn

    print("Không thể trích xuất triệu chứng từ Gemini.")
    return [], "Không thể trích xuất triệu chứng từ Gemini. Vui lòng thử lại sau."