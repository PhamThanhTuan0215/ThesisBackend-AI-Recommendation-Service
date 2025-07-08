# Sử dụng image Python nhẹ
FROM python:3.10-slim

# Tối ưu môi trường
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GEMINI_API_KEY=AIzaSyALbGBJIZK5pupluDlwHSK9YVGf0MHUSiU

# Tạo thư mục làm việc
WORKDIR /app

# Copy requirements trước để tận dụng cache
COPY requirements.txt ./

# Cài đặt các thư viện cần thiết
RUN apt-get update && apt-get install -y build-essential \
    && pip install --upgrade pip \
    && pip install -r requirements.txt \
    && apt-get remove -y build-essential && apt-get clean

# Chỉ copy các file code và dữ liệu cần thiết
COPY index.py ./
COPY suggestions.py ./
COPY benh_nhomthuoc_normalized.xlsx ./
COPY benh_trieu_chung_normalized.xlsx ./
COPY nhomthuoc_thuoc_normalized.xlsx ./
COPY corpus_index.faiss ./
COPY corpus_lines.csv ./

# Thiết lập cổng mặc định (Railway sẽ override bằng biến PORT)
ENV PORT=10000

# Chạy ứng dụng
CMD ["uvicorn", "index:app", "--host", "0.0.0.0", "--port", "${PORT}"]