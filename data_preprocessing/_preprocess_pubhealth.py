import os
import csv
import json

def load_pubhealth_tsv(path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def convert_pubhealth(entries, prefix="PH"):
    claims = []
    docs = []

    label_map = {
        "true": "SUPPORTED",
        "false": "REFUTED",
        "mixture": "NOT ENOUGH INFO",
        "unproven": "NOT ENOUGH INFO"
    }

    for row in entries:
        cid = row.get("claim_id") or row.get("id")
        claim_text = (row.get("claim") or "").strip()
        main_text = (row.get("main_text") or "").strip()
        label = (row.get("label") or "").strip().lower()

        # Some test rows might have missing label
        label_unified = label_map.get(label, None) if label else None

        claim_id = f"{prefix}_{cid}"
        doc_id = f"DOC_{claim_id}"

        claims.append({
            "claim_id": claim_id,
            "claim": claim_text,
            "label": label_unified,
            "evidence_doc_ids": [doc_id],
            "dataset": "PubHealth"
        })

        docs.append({
            "id": doc_id,
            "title": "Evidence from PubHealth",
            "content": main_text
        })

    return claims, docs

# === Paths ===
input_dir = "../data/claims/raw/PUBHEALTH/"
output_dir = "../data/claims/processed/PubHealth/"
os.makedirs(os.path.join(output_dir, "claims"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "docs"), exist_ok=True)

splits = ["train", "dev", "test"]

for split in splits:
    input_path = os.path.join(input_dir, f"{split}.tsv")
    out_claims_path = os.path.join(output_dir, "claims", f"PubHealth_{split}.jsonl")
    out_docs_path = os.path.join(output_dir, "docs", f"pubhealth_docs_{split}.jsonl")

    rows = load_pubhealth_tsv(input_path)
    claims, docs = convert_pubhealth(rows)

    save_jsonl(claims, out_claims_path)
    save_jsonl(docs, out_docs_path)

    print(f"✅ Processed PubHealth {split}: {len(claims)} claims, {len(docs)} docs.")
