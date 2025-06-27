import json
import os
import re
from typing import List, Dict, Set


# Helper function (copied from your original script for consistency)
def split_into_sentences(text: str) -> List[str]:
    """Splits text into sentences."""
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]


def load_all_sentences_from_jsonl(jsonl_filepath: str) -> Dict[str, List[str]]:
    """
    Loads all documents from a .jsonl file and splits their content into sentences.

    Args:
        jsonl_filepath (str): Path to the .jsonl file.

    Returns:
        Dict[str, List[str]]: A dictionary mapping document_id to a list of its sentences.
    """
    doc_sentences_map = {}
    if not os.path.exists(jsonl_filepath):
        print(f"Error: Original data file not found at {jsonl_filepath}")
        return doc_sentences_map

    with open(jsonl_filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                doc = json.loads(line.strip())
                doc_id = doc.get("id")
                content = doc.get("content")
                if not doc_id or not content:
                    print(f"Warning: Skipping line {line_num} due to missing 'id' or 'content' in {jsonl_filepath}")
                    continue
                doc_sentences_map[doc_id] = split_into_sentences(content)
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON on line {line_num} in {jsonl_filepath}")
            except Exception as e:
                print(f"An error occurred while processing line {line_num} in {jsonl_filepath}: {e}")
    return doc_sentences_map


def load_fact_source_sentences_from_kg(kg_json_filepath: str) -> Dict[str, Set[str]]:
    """
    Loads the knowledge graph and extracts all unique sentences from which facts were derived.

    Args:
        kg_json_filepath (str): Path to the knowledge graph JSON file.

    Returns:
        Dict[str, Set[str]]: A dictionary mapping document_id to a set of its sentences
                               that successfully produced facts.
    """
    fact_sentences_map = {}
    if not os.path.exists(kg_json_filepath):
        print(f"Error: Knowledge graph file not found at {kg_json_filepath}")
        return fact_sentences_map

    try:
        with open(kg_json_filepath, 'r', encoding='utf-8') as f:
            kg_data = json.load(f)  # Assuming the KG is a list of facts as per your script

        # If kg_data is a dict with a "knowledge_graph" key (as in the modified example)
        if isinstance(kg_data, dict) and "knowledge_graph" in kg_data:
            facts_list = kg_data["knowledge_graph"]
        # If kg_data is directly a list of facts (as in the original script's output structure)
        elif isinstance(kg_data, list):
            facts_list = kg_data
        else:
            print(
                f"Warning: Unexpected KG file structure in {kg_json_filepath}. Expected list or dict with 'knowledge_graph' key.")
            return fact_sentences_map

        for fact_entry in facts_list:
            doc_id = fact_entry.get("document_id")
            sentence = fact_entry.get("sentence")
            if doc_id and sentence:
                if doc_id not in fact_sentences_map:
                    fact_sentences_map[doc_id] = set()
                fact_sentences_map[doc_id].add(sentence.strip())
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in knowledge graph file {kg_json_filepath}")
    except Exception as e:
        print(f"An error occurred while processing {kg_json_filepath}: {e}")
    return fact_sentences_map


def find_skipped_sentences(
        all_doc_sentences: Dict[str, List[str]],
        fact_source_sentences: Dict[str, Set[str]]
) -> Dict[str, List[str]]:
    """
    Compares all sentences with fact-producing sentences to find skipped ones.

    Args:
        all_doc_sentences (Dict[str, List[str]]): All sentences per document_id.
        fact_source_sentences (Dict[str, Set[str]]): Fact-producing sentences per document_id.

    Returns:
        Dict[str, List[str]]: A dictionary mapping document_id to a list of its skipped sentences.
    """
    skipped_sentences_map = {}
    for doc_id, sentences_in_doc in all_doc_sentences.items():
        processed_sentences = fact_source_sentences.get(doc_id, set())
        skipped_for_this_doc = []
        for sentence in sentences_in_doc:
            if sentence.strip() not in processed_sentences:
                skipped_for_this_doc.append(sentence)

        if skipped_for_this_doc:  # Only add if there are skipped sentences
            skipped_sentences_map[doc_id] = skipped_for_this_doc
    return skipped_sentences_map


def process_files(original_jsonl_path: str, kg_json_path: str, output_skipped_path: str):
    """
    Main processing function to find and save skipped sentences.
    """
    print(f"🔍 Loading all sentences from: {original_jsonl_path}")
    all_sentences = load_all_sentences_from_jsonl(original_jsonl_path)
    if not all_sentences:
        print("No sentences loaded from original file. Exiting.")
        return

    print(f"📖 Loading fact source sentences from: {kg_json_path}")
    fact_sentences = load_fact_source_sentences_from_kg(kg_json_path)
    # No need to exit if fact_sentences is empty, as it just means all sentences were skipped.

    print("🔄 Comparing sentences to find skipped ones...")
    skipped_data = find_skipped_sentences(all_sentences, fact_sentences)

    if skipped_data:
        with open(output_skipped_path, 'w', encoding='utf-8') as outfile:
            json.dump(skipped_data, outfile, indent=2)
        print(f"✅ Skipped sentences saved to: {output_skipped_path}")
        for doc_id, sents in skipped_data.items():
            print(f"  📄 Document ID '{doc_id}' had {len(sents)} skipped sentences.")
    else:
        print("🎉 No skipped sentences found or no documents to process!")


if __name__ == "__main__":
    # --- Configuration ---
    # Path to the directory containing the original .jsonl files
    # This should match `input_root` from your main script, then dataset, then docs
    original_data_root = "../../data/claims/processed"

    # Path to the directory where your KG JSON files are stored
    # This should match `output_root` from your main script
    kg_output_root = "../../data/claims/kg_from_ref_docs"

    # Path where you want to save the skipped sentences reports
    skipped_reports_root = "../../data/skipped_sentence_reports"
    os.makedirs(skipped_reports_root, exist_ok=True)

    # Datasets to process (mirroring your main script, excluding skipped_datasets)
    # You might want to get this list dynamically if it changes often
    datasets_to_process = []
    all_datasets_in_input = os.listdir(original_data_root)
    skipped_datasets_config = ["CoVERT", "Fever" "PubHealth", "VitaminC"]#, "HealthVer", "Covid-Fact",

    for ds_name in all_datasets_in_input:
        if os.path.isdir(os.path.join(original_data_root, ds_name)) and ds_name not in skipped_datasets_config:
            datasets_to_process.append(ds_name)

    print(f"Found datasets to process: {datasets_to_process}")

    total_files_analyzed = 0
    # --- Processing Loop ---
    for dataset_name in datasets_to_process:
        original_dataset_docs_path = os.path.join(original_data_root, dataset_name, "docs")
        kg_dataset_path = os.path.join(kg_output_root, dataset_name)
        skipped_dataset_output_path = os.path.join(skipped_reports_root, dataset_name)
        os.makedirs(skipped_dataset_output_path, exist_ok=True)

        if not os.path.exists(original_dataset_docs_path):
            print(f"Warning: Original docs path not found for dataset '{dataset_name}': {original_dataset_docs_path}")
            continue
        if not os.path.exists(kg_dataset_path):
            print(f"Warning: KG output path not found for dataset '{dataset_name}': {kg_dataset_path}")
            continue

        print(f"\n--- Processing Dataset: {dataset_name} ---")

        for item_name in os.listdir(original_dataset_docs_path):
            if item_name.endswith(".jsonl"):
                original_file_path = os.path.join(original_dataset_docs_path, item_name)

                # Construct the corresponding KG file name
                # kg_{dataset_name}_{file_name.replace('.jsonl', '.json')}
                kg_file_name = f"kg_{dataset_name}_{item_name.replace('.jsonl', '.json')}"
                kg_file_path = os.path.join(kg_dataset_path, kg_file_name)

                output_skipped_filename = f"skipped_sentences_{item_name.replace('.jsonl', '.json')}"
                output_skipped_filepath = os.path.join(skipped_dataset_output_path, output_skipped_filename)

                print(f"\nAnalysing original: {original_file_path}")
                print(f"Against KG:       {kg_file_path}")
                print(f"Outputting to:    {output_skipped_filepath}")

                if not os.path.exists(kg_file_path):
                    print(f"Warning: KG file {kg_file_path} not found. Skipping analysis for {item_name}.")
                    continue

                process_files(original_file_path, kg_file_path, output_skipped_filepath)
                total_files_analyzed += 1

    print(f"\n--- Analysis Complete ---")
    print(f"Processed {total_files_analyzed} pairs of original/KG files.")
    print(f"Skipped sentence reports saved in: {skipped_reports_root}")