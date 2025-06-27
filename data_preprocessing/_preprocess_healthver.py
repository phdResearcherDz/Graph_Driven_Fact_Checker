import os
import csv
import json

def load_csv(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

def convert_healthver(entries, prefix="HV"):
    claims = []
    docs = []

    for row in entries:
        id_ = row["id"].strip()
        claim = row["claim"].strip()
        evidence = row["evidence"].strip()
        label = row["label"].strip().upper()  # Convert to SUPPORTED, REFUTED, etc.

        label_map = {
            "supports": "SUPPORTED",
            "refutes": "REFUTED",
            "neutral": "NOT ENOUGH INFO"
        }
        label_final = label_map.get(label.lower(), "NOT ENOUGH INFO")

        claim_id = f"{prefix}_{id_}"
        doc_id = f"DOC_{claim_id}"

        claims.append({
            "claim_id": claim_id,
            "claim": claim,
            "label": label_final,
            "evidence_doc_ids": [doc_id],
            "dataset": "HealthVer"
        })

        docs.append({
            "id": doc_id,
            "title": "Evidence from HealthVer",
            "content": evidence
        })

    return claims, docs

# === Paths ===
base_input = "../data/claims/raw/HealthVer/"
base_output = "../data/claims/processed/HealthVer/"
os.makedirs(os.path.join(base_output, "claims"), exist_ok=True)
os.makedirs(os.path.join(base_output, "docs"), exist_ok=True)

splits = ["train", "dev", "test"]
for split in splits:
    input_csv_path = os.path.join(base_input, f"{split}.csv")
    output_claims_path = os.path.join(base_output, "claims", f"HealthVer_{split}.jsonl")
    output_docs_path = os.path.join(base_output, "docs", f"healthver_docs_{split}.jsonl")

    data = load_csv(input_csv_path)
    claims, docs = convert_healthver(data, prefix=f"HV_{split.upper()}")

    save_jsonl(claims, output_claims_path)
    save_jsonl(docs, output_docs_path)

    print(f"✅ Processed HealthVer {split}: {len(claims)} claims, {len(docs)} docs.")
