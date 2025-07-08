import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from collections import defaultdict

# 1. Load model, FAISS index, corpus và file mapping
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
index = faiss.read_index("corpus_index.faiss")
corpus = pd.read_csv("corpus_lines.csv", header=None)[0].tolist()

df_symptom = pd.read_excel("benh_trieu_chung_normalized.xlsx")
df_disease = pd.read_excel("benh_nhomthuoc_normalized.xlsx")
df_drug = pd.read_excel("nhomthuoc_thuoc_normalized.xlsx")

# 2. Hàm tìm kiếm top câu tương tự từ FAISS + tính điểm cosine
def search_symptom(symptom_input, top_k=30):
    embedding = model.encode([symptom_input])
    D, I = index.search(embedding, top_k)
    results = []
    for idx in I[0]:
        matched_line = corpus[idx]
        if "Triệu chứng:" in matched_line and "→ Có thể liên quan đến bệnh:" in matched_line:
            parts = matched_line.split("→")
            matched_symptom = parts[0].replace("Triệu chứng:", "").strip()
            score = cosine_similarity(model.encode([symptom_input]), model.encode([matched_symptom]))[0][0]
            results.append((matched_line, matched_symptom, round(score, 3)))
    return results

# 3. Giao nhau các bệnh từ nhiều triệu chứng (hoặc chọn bệnh khớp nhiều triệu chứng nhất)
def suggest_diseases(symptoms, min_score=0.6, top_k=3):
    all_disease_sets = []
    matched_symptom_logs = []
    all_matched_diseases = []

    for symptom in symptoms:
        matches = search_symptom(symptom)
        diseases = set()
        for matched_line, matched_symptom, score in matches:
            if (score < min_score): # nếu điểm cosine nhỏ hơn 0.7 thì bỏ qua (không đủ độ tin cậy)
                continue
            parts = matched_line.split("→")
            disease = parts[1].replace("Có thể liên quan đến bệnh:", "").strip()
            matched_symptom_logs.append({
                "input_symptom": symptom,
                "matched_symptom": matched_symptom,
                "disease": disease,
                "score": score
            })
            diseases.add(disease)
            all_matched_diseases.append(disease)
        if diseases:
            all_disease_sets.append(diseases)

    is_perfect_match_disease = False

    if not all_disease_sets:
        return set(), matched_symptom_logs, is_perfect_match_disease

    # Giao nhau
    common_diseases = set.intersection(*all_disease_sets)
    if common_diseases:
        is_perfect_match_disease = True
        if len(common_diseases) > top_k:
            # chỉ lấy ra 3 bệnh
            common_diseases = set(list(common_diseases)[:top_k])
        return common_diseases, matched_symptom_logs, is_perfect_match_disease

    # Không có giao nhau → fallback: chọn bệnh xuất hiện nhiều nhất
    fallback_counter = Counter(all_matched_diseases)
    most_common = fallback_counter.most_common(top_k) # nếu không có bệnh nào phù hợp hoàn toàn thì lấy top 3 bệnh gần đúng nhất
    fallback_diseases = set(disease for disease, count in most_common)
    return fallback_diseases, matched_symptom_logs, is_perfect_match_disease

# 4. Gợi ý nhóm thuốc và tên thuốc từ bệnh
def suggest_meds_and_drugs(disease):
    suggestions = []
    med_groups = df_disease[df_disease["Bệnh"] == disease]["Nhóm thuốc"].unique()
    for med in med_groups:
        suggestion = {
            "med_group": med,
            "drugs": []
        }
        drugs = df_drug[df_drug["Nhóm thuốc"] == med]["Tên thuốc"].unique()
        for drug in drugs:
            suggestion["drugs"].append(drug)
        suggestions.append(suggestion)
    return suggestions

# 5. Hàm tổng hợp tất cả dữ liệu về gợi ý thuốc
def get_all_data_suggestions(symptoms_input, min_score=0.6, top_k=3):

    # lấy triệu chứng từ user input
    symptoms = [] # data thứ 1
    matched_logs = [] # data thứ 2
    matched_symptoms = [] # data thứ 3
    diseases_result = [] # data thứ 4
    is_perfect_match_disease = False

    symptoms = symptoms_input

    # giao nhau các bệnh từ nhiều triệu chứng
    common_diseases, matched_symptom_logs, is_perfect_match_disease = suggest_diseases(symptoms, min_score, top_k)

    if matched_symptom_logs:
        shown_pairs = set()
        for log in matched_symptom_logs:
            user_input_symptom = log["input_symptom"]
            matched_symptom = log["matched_symptom"]
            score = log["score"]

            pair_key = (user_input_symptom, matched_symptom)
            if pair_key not in shown_pairs:
                matched_logs.append(f"Bạn nhập: '{user_input_symptom}' → Khớp gần với: '{matched_symptom}' (score: {score * 100:.2f}%)")
                matched_symptoms.append(matched_symptom)
                shown_pairs.add(pair_key)

    if common_diseases:
        # Gom ánh xạ: disease → {input_symptom → (matched_symptom, score cao nhất)}
        disease_to_input_symptom_map = defaultdict(dict)

        for log in matched_symptom_logs:
            disease = log["disease"]
            input_symptom = log["input_symptom"]
            matched_symptom = log["matched_symptom"]
            score = log["score"]

            if disease in common_diseases:
                # Nếu chưa có hoặc điểm mới cao hơn thì ghi đè
                if (input_symptom not in disease_to_input_symptom_map[disease] or 
                    score > disease_to_input_symptom_map[disease][input_symptom][1]):
                    disease_to_input_symptom_map[disease][input_symptom] = (matched_symptom, score)

        # Sắp xếp theo số triệu chứng gốc khớp được
        sorted_diseases = sorted(
            disease_to_input_symptom_map.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )

        for disease, input_map in sorted_diseases:
            formatted = [
                f"{matched_symptom} (từ '{input_symptom}' {score*100:.2f}%)"
                for input_symptom, (matched_symptom, score) in input_map.items()
            ]
            disase_result = {
                "disease": disease,
                "note": f"{len(formatted)} triệu chứng khớp gồm: {', '.join(formatted)}",
                "suggestions": suggest_meds_and_drugs(disease)
            }
            diseases_result.append(disase_result)

    data = {
        "min_score": min_score,
        "top_k": len(diseases_result),
        "symptoms_input": symptoms,
        "matched_logs": matched_logs,
        "matched_symptoms": matched_symptoms,
        "is_perfect_match_disease": is_perfect_match_disease,
        "diseases_result": diseases_result,
    }

    return data

# 6. CLI
if __name__ == "__main__":
    while True:
        user_input = input("\n🩺 Nhập triệu chứng (cách nhau bởi 'và'): ").strip()
        if not user_input:
            break

        # Xử lý đầu vào
        symptoms_input = [s.strip() for s in user_input.split("và") if s.strip()]

        min_score = 0.6 # điểm tin cậy
        top_k = 3 # số bệnh gần đúng nhất

        data = get_all_data_suggestions(symptoms_input, min_score, top_k)
        symptoms_input = data["symptoms_input"]
        matched_logs = data["matched_logs"]
        matched_symptoms = data["matched_symptoms"]
        diseases_result = data["diseases_result"]
        is_perfect_match_disease = data["is_perfect_match_disease"]
        min_score = data["min_score"]
        top_k = data["top_k"]

        # in ra các triệu chứng đang xử lý (data 1)
        print(f"\n🔍 Đang xử lý {len(symptoms_input)} triệu chứng:")
        for symptom in symptoms_input:
            print(f"• {symptom}")

        # in ra các dòng ánh xạ triệu chứng (data 2)
        if (matched_logs):
            print("\n🧠 Ánh xạ triệu chứng gần đúng:")
            for log in matched_logs:
                print(f"• {log}")
        else:
            print("\n⚠️ Không tìm thấy triệu chứng phù hợp.")

        # in ra các triệu chứng phù hợp ánh xạ được (data 3)
        if (matched_symptoms):
            print("\nCác triệu chứng phù hợp ánh xạ được:")
            for matched_symptom in matched_symptoms:
                print(f"• {matched_symptom}")

        if not diseases_result:
            print("\n⚠️ Không tìm thấy bệnh phù hợp với tất cả triệu chứng.")
        else:
            # in kết quả các bệnh dự đoán
            if (is_perfect_match_disease):
                print("\n📋 Dự đoán bệnh phù hợp:")
            else:
                print("\n📋 Các bệnh dự đoán gần đúng (xếp theo số triệu chứng khớp):")

            for disease_result in diseases_result:
                print(f"✅ {disease_result['disease']}") 
                print(f"→ {disease_result['note']}")

            # in ra các gợi ý nhóm thuốc và sản phẩm
            print("\n💊 Gợi ý nhóm thuốc và sản phẩm:")
            for disease_result in diseases_result:
                print(f"• Bệnh: {disease_result['disease']}")
                for suggestion in disease_result["suggestions"]:
                    print(f"  → {suggestion['med_group']} → Sản phẩm gợi ý: {', '.join(suggestion['drugs'])}")