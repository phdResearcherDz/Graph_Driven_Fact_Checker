import json
import os
from glob import glob

def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def convert_fever_claims(claims, dataset_split="train"):
    formatted_claims = []
    evidence_ids = set()

    for claim in claims:
        claim_id = f"FEVER_{claim['id']}"
        label = claim.get("label", "NOT ENOUGH INFO")
        all_evidence_sets = claim.get("evidence", [])
        doc_ids = []

        for evidence_set in all_evidence_sets:
            for e in evidence_set:
                # Skip unlinked evidence (from NOT ENOUGH INFO claims)
                if len(e) >= 4 and e[2] is not None and e[3] is not None:
                    wiki_title = e[2]
                    sentence_id = e[3]
                    doc_id = f"DOC_{wiki_title}_{sentence_id}"
                    doc_ids.append(doc_id)
                    evidence_ids.add((wiki_title, sentence_id))

        formatted_claims.append({
            "claim_id": claim_id,
            "claim": claim["claim"],
            "label": label,
            "evidence_doc_ids": doc_ids,
            "dataset": "FEVER"
        })

    return formatted_claims, evidence_ids

def build_wiki_sentence_lookup(wiki_dir):
    lookup = {}
    for filepath in glob(os.path.join(wiki_dir, "*.jsonl")):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    title = item["id"]
                    for line_str in item["lines"].split("\n"):
                        if "\t" in line_str:
                            sid, text = line_str.split("\t", 1)
                            lookup[(title, int(sid))] = text.strip()
                except Exception as e:
                    print(f"⚠️ Skipping malformed line in {filepath}: {e}")
    return lookup

def generate_fever_docs(wiki_sentence_lookup, evidence_ids):
    docs = []
    for (title, sent_id) in evidence_ids:
        sentence = wiki_sentence_lookup.get((title, int(sent_id)), "")
        if sentence:  # Avoid saving empty docs
            doc = {
                "id": f"DOC_{title}_{sent_id}",
                "title": title,
                "content": sentence
            }
            docs.append(doc)
    return docs

# === CONFIGURATION ===
dataset_split = "test"  # or "dev"
base_input = "../data/claims/raw/Fever"
base_output = "../data/claims/processed/Fever"
wiki_dir = os.path.join(base_input, "wiki-pages")

input_claims_path = os.path.join(base_input, f"{dataset_split}.jsonl")
output_claims_path = os.path.join(base_output, f"FEVER_{dataset_split}.jsonl")
output_docs_path = os.path.join(base_output, f"fever_docs_{dataset_split}.jsonl")

# === PROCESSING ===
claims = load_jsonl(input_claims_path)
formatted_claims, evidence_ids = convert_fever_claims(claims, dataset_split=dataset_split)
wiki_sentence_lookup = build_wiki_sentence_lookup(wiki_dir)
fever_docs = generate_fever_docs(wiki_sentence_lookup, evidence_ids)

# === SAVE ===
os.makedirs(os.path.dirname(output_claims_path), exist_ok=True)
save_jsonl(formatted_claims, output_claims_path)
save_jsonl(fever_docs, output_docs_path)

print(f"✅ FEVER {dataset_split} set processed. {len(formatted_claims)} claims, {len(fever_docs)} evidence docs.")
