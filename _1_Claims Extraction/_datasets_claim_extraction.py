import os
import json
from _claim_extractor_v2 import extract_from_text  # Your enhanced extractor
import re

def clean_claim_text(text: str) -> str:
    # Remove @mentions and #hashtags
    text = re.sub(r"[@#]\w+", "", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove stray brackets, emojis, and non-ASCII chars
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

# Base input and output folders
input_root = "../../data/claims/processed"
output_root = "../../data/claims/claim_extracted"
os.makedirs(output_root, exist_ok=True)

skiped_datasets = ["CoVERT", "Fever", "Covid-Fact","HealthVer","PubHealth","SciFact"]
# Process each dataset folder
for dataset_name in os.listdir(input_root):
    dataset_path = os.path.join(input_root, dataset_name)

    # Skip if not a directory or explicitly excluded
    if not os.path.isdir(dataset_path) or dataset_name in skiped_datasets:
        continue

    claims_path = os.path.join(dataset_path, "claims")
    if not os.path.exists(claims_path):
        print(f"⚠️ Skipping: No 'claims/' folder in {dataset_name}")
        continue

    # Prepare output folder
    dataset_output_folder = os.path.join(output_root, dataset_name)
    os.makedirs(dataset_output_folder, exist_ok=True)

    for split_file in ["train.jsonl", "dev.jsonl", "test.jsonl"]:
        input_file_path = os.path.join(claims_path, split_file)
        if not os.path.exists(input_file_path):
            print(f"ℹ️ {split_file} not found in {dataset_name}")
            continue

        enriched_claims = []

        with open(input_file_path, "r", encoding="utf-8") as infile:
            for line in infile:
                try:
                    entry = json.loads(line)
                    claim_text = entry.get("claim", "").strip()
                    claim_id = entry.get("claim_id", "").strip()

                    if not claim_text:
                        continue
                    clean_claim = clean_claim_text(claim_text)
                    # Extract medical facts
                    result = extract_from_text(clean_claim)
                    facts = result.get("facts", [])

                    enriched_claims.append({
                        "claim_id": claim_id,
                        "claim": clean_claim,
                        "facts": facts
                    })

                except Exception as e:
                    print(f"❌ Error in {dataset_name}/{split_file}: {e}")

        # Save output
        output_file_path = os.path.join(dataset_output_folder, split_file)
        with open(output_file_path, "w", encoding="utf-8") as outfile:
            for enriched in enriched_claims:
                outfile.write(json.dumps(enriched) + "\n")

        print(f"✅ Finished: {dataset_name}/{split_file} → {output_file_path}")
