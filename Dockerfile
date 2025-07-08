# Sử dụng image Python nhẹ
FROM python:3.10-slim

# Tối ưu môi trường
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Tạo thư mục làm việc
WORKDIR /app

# Cài đặt các thư viện cần thiết
COPY requirements.txt .
RUN apt-get update && apt-get install -y build-essential \
    && pip install --upgrade pip \
    && pip install -r requirements.txt \
    && apt-get remove -y build-essential && apt-get clean

# Copy toàn bộ mã nguồn và dữ liệu vào image
COPY . .

# Thiết lập cổng mặc định (Railway sẽ override bằng biến PORT)
ENV PORT=10000

# Chạy ứng dụng
CMD ["uvicorn", "index:app", "--host", "0.0.0.0", "--port", "10000"]