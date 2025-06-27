import csv
import json
import os

def load_tsv(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = []
        for row in reader:
            normalized = {k.strip().lower(): v for k, v in row.items()}
            rows.append(normalized)
        return rows

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def convert_covid_fact(tsv_rows, prefix="COVID"):
    claims = []
    docs = []

    for i, row in enumerate(tsv_rows):
        claim_id = f"{prefix}_{i}"
        doc_id = f"DOC_{claim_id}"

        label_map = {
            "entailment": "SUPPORTED",
            "not_entailment": "REFUTED",
            "neutral": "NOT ENOUGH INFO"  # optional
        }
        label_raw = row.get("entailment", row.get("label", "neutral")).strip().lower()
        mapped_label = label_map.get(label_raw, "NOT ENOUGH INFO")

        claims.append({
            "claim_id": claim_id,
            "claim": row["sentence2"].strip(),
            "label": mapped_label,
            "evidence_doc_ids": [doc_id],
            "dataset": "COVID-Fact"
        })

        docs.append({
            "id": doc_id,
            "title": "Evidence from COVID-Fact",
            "content": row["sentence1"].strip()
        })

    return claims, docs

# === Paths ===
base_input = "../data/claims/raw/Covid-Fact/RTE-covidfact/"
base_output = "../data/claims/processed/Covid-Fact/"
os.makedirs(os.path.join(base_output, "claims"), exist_ok=True)
os.makedirs(os.path.join(base_output, "docs"), exist_ok=True)

splits = ["train", "dev", "test"]
for split in splits:
    input_path = os.path.join(base_input, f"{split}.tsv")
    output_claims_path = os.path.join(base_output, "claims", f"COVID_{split}.jsonl")
    output_docs_path = os.path.join(base_output, "docs", f"COVID_docs_{split}.jsonl")

    tsv_data = load_tsv(input_path)
    claims, docs = convert_covid_fact(tsv_data, prefix=f"COVID_{split.upper()}")

    save_jsonl(claims, output_claims_path)
    save_jsonl(docs, output_docs_path)

    print(f"✅ Processed {split}: {len(claims)} claims, {len(docs)} documents.")
