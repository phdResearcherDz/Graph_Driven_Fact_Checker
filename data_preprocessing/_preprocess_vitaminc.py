import json
import os

def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def convert_vitaminC(entries, prefix="VITC"):
    claims = []
    docs = []

    for item in entries:
        uid = item["unique_id"]
        doc_id = f"DOC_{prefix}_{uid}"
        claim_id = f"{prefix}_{uid}"

        claims.append({
            "claim_id": claim_id,
            "claim": item["claim"].strip(),
            "label": item["label"].strip().upper(),
            "evidence_doc_ids": [doc_id],
            "dataset": "VitaminC"
        })

        docs.append({
            "id": doc_id,
            "title": item.get("page", "VitaminC Evidence"),
            "content": item["evidence"].strip()
        })

    return claims, docs

# === Paths ===
base_input = "../data/claims/raw/VitaminC/"
base_output = "../data/claims/processed/VitaminC/"
os.makedirs(os.path.join(base_output, "claims"), exist_ok=True)
os.makedirs(os.path.join(base_output, "docs"), exist_ok=True)

splits = ["train", "dev", "test"]
for split in splits:
    input_path = os.path.join(base_input, f"{split}.jsonl")
    output_claims_path = os.path.join(base_output, "claims", f"VitaminC_{split}.jsonl")
    output_docs_path = os.path.join(base_output, "docs", f"vitaminc_docs_{split}.jsonl")

    data = load_jsonl(input_path)
    claims, docs = convert_vitaminC(data, prefix=f"VITC_{split.upper()}")

    save_jsonl(claims, output_claims_path)
    save_jsonl(docs, output_docs_path)

    print(f"✅ Processed VitaminC {split}: {len(claims)} claims, {len(docs)} docs.")
