import csv
import json
import os

def load_tsv(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def convert_covert(entries, prefix="COVERT"):
    claims, docs = [], []

    for entry in entries:
        tweet_id = entry["tweet_id"]
        tweet_text = entry["tweet_text"].strip()
        claim_text = (
            entry.get("implicit_claim", "").strip()
            if entry.get("implicit_claim", "").strip()
            else tweet_text  # fallback
        )

        label_raw = entry.get("claim", "").strip().lower()
        if label_raw == "true":
            label = "SUPPORTED"
        elif label_raw == "false":
            label = "REFUTED"
        else:
            label = "NOT ENOUGH INFO"

        claim_id = f"{prefix}_{tweet_id}"
        doc_id = f"DOC_{claim_id}"

        claims.append({
            "claim_id": claim_id,
            "claim": claim_text,
            "label": label,
            "evidence_doc_ids": [doc_id],
            "dataset": "CoVERT"
        })

        docs.append({
            "id": doc_id,
            "title": "Tweet Evidence",
            "content": tweet_text
        })

    return claims, docs

# === Paths ===
base_input = "../data/claims/raw/CoVERT/corpus/"
base_output = "../data/claims/processed/CoVERT/"
os.makedirs(os.path.join(base_output, "claims"), exist_ok=True)
os.makedirs(os.path.join(base_output, "docs"), exist_ok=True)

splits = ["train", "dev", "test"]
for split in splits:
    input_path = os.path.join(base_input, f"{split}.tsv")
    output_claims_path = os.path.join(base_output, "claims", f"{split}.jsonl")
    output_docs_path = os.path.join(base_output, "docs", f"{split}.jsonl")

    tsv_data = load_tsv(input_path)
    claims, docs = convert_covert(tsv_data, prefix=f"COVERT_{split.upper()}")

    save_jsonl(claims, output_claims_path)
    save_jsonl(docs, output_docs_path)

    print(f"✅ Processed CoVERT {split}: {len(claims)} claims, {len(docs)} documents.")
