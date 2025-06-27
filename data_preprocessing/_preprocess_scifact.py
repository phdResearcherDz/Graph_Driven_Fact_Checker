import json
import os

def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_jsonl(data_list, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data_list:
            f.write(json.dumps(item) + "\n")

def convert_scifact_claims(claims, dataset_split="train"):
    formatted_claims = []
    used_doc_ids = set()

    for claim in claims:
        claim_id = f"SF_{claim['id']}"
        claim_text = claim["claim"]
        evidence = claim.get("evidence", {})
        cited_doc_ids = claim.get("cited_doc_ids", [])
        label = "NOT ENOUGH INFO"
        evidence_doc_ids = set()

        if evidence:
            labels = set()
            for doc_id, evs in evidence.items():
                evidence_doc_ids.add(str(doc_id))
                for item in evs:
                    if item["label"] == "SUPPORT":
                        labels.add("SUPPORTED")
                    elif item["label"] == "CONTRADICT":
                        labels.add("REFUTED")
            if "SUPPORTED" in labels and "REFUTED" in labels:
                label = "NOT ENOUGH INFO"
            elif "REFUTED" in labels:
                label = "REFUTED"
            elif "SUPPORTED" in labels:
                label = "SUPPORTED"
        else:
            evidence_doc_ids.update(map(str, cited_doc_ids))

        used_doc_ids.update(evidence_doc_ids)

        formatted_claim = {
            "claim_id": claim_id,
            "claim": claim_text,
            "label": label if dataset_split != "test" else None,
            "evidence_doc_ids": [f"DOC{doc_id}" for doc_id in sorted(evidence_doc_ids)] if dataset_split != "test" else [],
            "dataset": "SciFact"
        }

        formatted_claims.append(formatted_claim)

    return formatted_claims, used_doc_ids

def extract_evidence_docs(doc_ids, corpus):
    doc_id_map = {str(doc['doc_id']): doc for doc in corpus}
    docs = []

    for doc_id in doc_ids:
        doc = doc_id_map.get(doc_id)
        if doc:
            docs.append({
                "id": f"DOC{doc['doc_id']}",
                "title": doc.get("title", ""),
                "content": " ".join(doc.get("abstract", []))
            })
    return docs

def restructure_corpus(corpus):
    """Restructure the entire corpus to the desired format."""
    restructured_corpus = []
    for doc in corpus:
        restructured_doc = {
            "id": f"DOC{doc['doc_id']}",
            "title": doc.get("title", ""),
            "content": " ".join(doc.get("abstract", []))
        }
        restructured_corpus.append(restructured_doc)
    return restructured_corpus
# === Paths ===
input_dir = "../data/claims/raw/SciFact"
output_dir = "../data/claims/processed/SciFact"
os.makedirs(os.path.join(output_dir, "claims"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "docs"), exist_ok=True)

# === Load Inputs ===
corpus = load_jsonl(os.path.join(input_dir, "corpus.jsonl"))
train_claims = load_jsonl(os.path.join(input_dir, "claims_train.jsonl"))
dev_claims = load_jsonl(os.path.join(input_dir, "claims_dev.jsonl"))
test_claims = load_jsonl(os.path.join(input_dir, "claims_test.jsonl"))

# === Convert and Save Train ===
train_data, train_doc_ids = convert_scifact_claims(train_claims, dataset_split="train")
train_docs = extract_evidence_docs(train_doc_ids, corpus)
save_jsonl(train_data, os.path.join(output_dir, "claims", "train.jsonl"))
save_jsonl(train_docs, os.path.join(output_dir, "docs", "train.jsonl"))

# === Convert and Save Dev ===
dev_data, dev_doc_ids = convert_scifact_claims(dev_claims, dataset_split="dev")
dev_docs = extract_evidence_docs(dev_doc_ids, corpus)
save_jsonl(dev_data, os.path.join(output_dir, "claims", "dev.jsonl"))
save_jsonl(dev_docs, os.path.join(output_dir, "docs", "dev.jsonl"))

# === Convert and Save Test ===
test_data, _ = convert_scifact_claims(test_claims, dataset_split="test")
save_jsonl(test_data, os.path.join(output_dir, "claims", "test.jsonl"))

# === Restructure and Save Entire Corpus ===
restructured_corpus = restructure_corpus(corpus)
save_jsonl(restructured_corpus, os.path.join(output_dir, "docs", "corpus.jsonl"))

print("✅ SciFact train, dev, and test JSONL processing complete.")
