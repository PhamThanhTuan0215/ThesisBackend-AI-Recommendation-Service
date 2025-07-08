from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd

# Load corpus
with open("corpus_khuyen_nghi.txt", encoding="utf-8") as f:
    corpus = [line.strip() for line in f if line.strip()]

# Dùng mô hình nhúng
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Tạo embedding
embeddings = model.encode(corpus, convert_to_numpy=True)

# Tạo FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Lưu index và corpus
faiss.write_index(index, "corpus_index.faiss")
pd.Series(corpus).to_csv("corpus_lines.csv", index=False, header=False)
