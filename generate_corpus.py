import pandas as pd

# Đọc file đã chuẩn hóa
df = pd.read_excel("data_normalized_final.xlsx")

# Tạo từng câu corpus từ mỗi dòng
corpus = df.apply(
    lambda row: f"Triệu chứng: {row.symptom} → Có thể liên quan đến bệnh: {row.disease} → Gợi ý nhóm thuốc: {row.medication} → Sản phẩm: {row.drug}",
    axis=1
)

# Xuất ra file text hoặc csv
# corpus.to_csv("corpus_khuyen_nghi.csv", index=False, header=False, encoding="utf-8")
corpus.to_csv("corpus_khuyen_nghi.txt", index=False, header=False, encoding="utf-8")
