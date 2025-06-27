import os
import json
import re
from collections import defaultdict
import numpy as np
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
import logging
import faiss  # Explicit import
import torch

# ------------------ CONFIG ------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EXTRACTION_ROOT = Path("../../../data/claims/claim_extracted")
PROCESSED_ROOT = Path("../../../data/claims/processed")
# Adjusted output root, ensure it aligns with your experiments
OUTPUT_ROOT = Path("../../../data/retrievals/docs_kg_structure_filtered")  # Changed output folder
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)  # Create if not exists

# ----- IMPORTANT: UPDATE THESE PATHS -----
INDEX_DIRs = [
    {
        "name": "Covid-Fact",
        "path": Path(
            "../../../data/Primary Evidance Source/_1_doc_as_kg_faiss_indexes_modernbert/Covid-Fact_triplets_flatL2"
        ),
        "max_distance": 1.0
    },
    {
        "name": "HealthVer",
        "path": Path(
            "../../../data/Primary Evidance Source/_1_doc_as_kg_faiss_indexes_modernbert/HealthVer_triplets_flatL2"
        ),
        "max_distance": 1.0
    },
    {
        "name": "SciFact",
        "path": Path(
            "../../../data/Primary Evidance Source/_1_doc_as_kg_faiss_indexes_modernbert/SciFact_triplets_flatL2"
        ),
        "max_distance": 1.0
    }
]

MODEL_NAME = "lightonai/modernbert-embed-large"
TOP_K = 50
BATCH_SIZE = 100  # Reduced for testing, adjust as needed


# ------------------ HELPERS (Unchanged) ------------------

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r"[@#]\w+", "", text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fact_triplet_to_query_string(fact_data: dict) -> str:
    subj = clean_text(fact_data.get('subject', ''))
    pred = clean_text(fact_data.get('predicate', ''))
    obj = clean_text(fact_data.get('object', ''))

    if not subj and not pred and not obj:
        return ""
    subj = subj if subj else "unknown"
    pred = pred if pred else "unknown"
    obj = obj if obj else "unknown"
    return f"{subj} [REL] {pred} [TAIL] {obj}"


# ------------------ SEARCH FUNCTION (Unchanged from your provided version) ------------------
def search_faiss_indexes(vector_stores, queries, top_k=TOP_K, max_distances_map=None):
    all_results_per_query_idx = {}

    if not queries:
        return all_results_per_query_idx

    try:
        embedding_function = vector_stores[next(iter(vector_stores))].embeddings
        query_embeddings = embedding_function.embed_documents(queries)
        query_embeddings_np = np.array(query_embeddings, dtype=np.float32)
    except Exception as e:
        logger.error(f"Error embedding queries: {e}", exc_info=True)
        for i in range(len(queries)):
            all_results_per_query_idx[i] = {index_name: [] for index_name in vector_stores}
        return all_results_per_query_idx

    for i, query_str in enumerate(queries):
        all_results_per_query_idx[i] = {}
        if not query_str:
            logger.debug(f"Skipping empty query string at flat index {i}.")
            for index_name in vector_stores.keys():
                all_results_per_query_idx[i][index_name] = []
            continue

        query_vector_np_2d = query_embeddings_np[i:i + 1]

        for index_name, vector_store in vector_stores.items():
            current_results_for_index = []
            try:
                if not vector_store.index.is_trained:
                    logger.warning(
                        f"FAISS index '{index_name}' is not trained. Skipping search.")
                    all_results_per_query_idx[i][index_name] = []
                    continue

                distances, faiss_indices = vector_store.index.search(query_vector_np_2d, k=top_k)
                current_max_distance = max_distances_map.get(index_name, float('inf'))

                for j in range(faiss_indices.shape[1]):
                    pos = faiss_indices[0, j]
                    dist = distances[0, j]

                    if pos == -1: break
                    if dist <= current_max_distance:
                        doc_id_in_docstore = vector_store.index_to_docstore_id.get(int(pos))
                        if doc_id_in_docstore:
                            doc = vector_store.docstore.search(doc_id_in_docstore)
                            if doc:
                                current_results_for_index.append((doc, float(dist)))
                            else:
                                logger.warning(
                                    f"Doc for ID '{doc_id_in_docstore}' (FAISS pos {pos}) not found in docstore for index {index_name}.")
                        else:
                            logger.warning(
                                f"FAISS position {pos} not in index_to_docstore_id map for index {index_name}.")
                all_results_per_query_idx[i][index_name] = current_results_for_index
            except Exception as e:
                logger.error(f"Error searching FAISS index {index_name} for query (idx {i}) '{query_str[:70]}...': {e}",
                             exc_info=True)
                all_results_per_query_idx[i][index_name] = []
    return all_results_per_query_idx


# ------------------ SCORE ANALYSIS FUNCTION (Unchanged) ------------------
def analyze_scores(scores, split_name, dataset_name, index_name):
    if not scores:
        logger.info(f"No scores to analyze for {dataset_name}/{split_name} ({index_name})")
        return 1.0  # Default to a high threshold if no scores
    scores_array = np.array(scores)
    # Ensure array is not empty before percentile calculation
    if scores_array.size == 0:
        logger.warning(f"Empty scores array for {dataset_name}/{split_name} ({index_name}). Defaulting P10 threshold.")
        return 1.0
    p10_score = float(np.percentile(scores_array, 10))
    return p10_score


# ------------------ MAIN FUNCTION (Updated with Filtering Logic) ------------------
def main():
    logger.info("Initializing embedding model for search...")
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name=MODEL_NAME,
            model_kwargs={"device": 0 if torch.cuda.is_available() else -1, "trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True},
            cache_folder="../../models/embeddings/")
        logger.info(f"Embedding model initialized. Device inferred from model_kwargs.")
    except Exception as e:
        logger.error(f"Error initializing embedding model for search: {e}", exc_info=True)
        return

    all_available_vector_stores = {}
    all_available_max_distances_map = {}
    logger.info("Loading ALL FAISS indices...")
    for index_info in INDEX_DIRs:
        index_name = index_info["name"]
        index_path_str = str(index_info["path"])
        if not index_info["path"].exists():
            logger.warning(f"Index directory {index_path_str} does not exist. Skipping '{index_name}'.")
            continue
        try:
            logger.info(f"Loading FAISS index '{index_name}' from {index_path_str}")
            vector_store = FAISS.load_local(
                folder_path=index_path_str,
                embeddings=embedding_model,
                allow_dangerous_deserialization=True,
            )
            all_available_vector_stores[index_name] = vector_store
            all_available_max_distances_map[index_name] = index_info.get("max_distance", float('inf'))
            logger.info(
                f"Index '{index_name}' loaded. Max distance: {all_available_max_distances_map[index_name]}. Index type: {type(vector_store.index)}")
        except Exception as e:
            logger.error(f"Error loading FAISS index '{index_name}': {e}", exc_info=True)
            continue

    if not all_available_vector_stores:
        logger.error("No FAISS indices were loaded. Exiting.")
        return
    logger.info(f"Successfully loaded indices: {list(all_available_vector_stores.keys())}")

    for dataset_folder in EXTRACTION_ROOT.iterdir():
        if not dataset_folder.is_dir(): continue
        dataset_name = dataset_folder.name
        logger.info(f"\n--- Processing dataset: {dataset_name} (Hybrid Search: Facts or Claim Text) ---")

        targeted_vector_store_dict = {}
        targeted_max_distances_map = {}
        if dataset_name in all_available_vector_stores:
            targeted_vector_store_dict[dataset_name] = all_available_vector_stores[dataset_name]
            targeted_max_distances_map[dataset_name] = all_available_max_distances_map[dataset_name]
            logger.info(f"Targeting FAISS index '{dataset_name}' for claims from '{dataset_name}' dataset.")
        else:
            logger.warning(
                f"No FAISS index named '{dataset_name}' configured for claims from this dataset. Will only use claim_text against available indexes or skip if none match.")
            # If we want to search against ALL loaded indexes regardless of dataset name match:
            # targeted_vector_store_dict = all_available_vector_stores
            # targeted_max_distances_map = all_available_max_distances_map
            # logger.info(f"No FAISS index named '{dataset_name}'. Claims from this dataset will be searched against ALL loaded indices.")
            # For now, let's stick to the original logic of requiring a matching index name.
            if not targeted_vector_store_dict:  # if still empty
                logger.warning(f"No suitable FAISS index found for dataset '{dataset_name}'. Skipping.")
                continue

        for split_filename in ["dev.jsonl", "test.jsonl", "train.jsonl"]:
            llm_extracted_claims_path = dataset_folder / split_filename
            reference_claims_path = PROCESSED_ROOT / dataset_name / "claims" / split_filename
            if not llm_extracted_claims_path.exists():  # Only llm_extracted needed to proceed
                logger.warning(
                    f"Skipping {split_filename} for {dataset_name} (missing LLM extracted file: {llm_extracted_claims_path})")
                continue

            reference_data_map = {}
            if reference_claims_path.exists():
                try:
                    with open(reference_claims_path, "r", encoding="utf-8") as ref_file:
                        for line in ref_file:
                            try:
                                ref_item = json.loads(line)
                                reference_data_map[ref_item["claim_id"]] = ref_item
                            except (json.JSONDecodeError, KeyError) as e:
                                logger.warning(
                                    f"Malformed JSON or missing 'claim_id' in {reference_claims_path}: {line.strip()} - Error: {e}")
                except Exception as e:
                    logger.error(f"Error loading ref data from {reference_claims_path}: {e}")
            else:
                logger.warning(
                    f"Reference claims file not found: {reference_claims_path}. Proceeding without reference data.")

            current_output_dir = OUTPUT_ROOT / dataset_name
            current_output_dir.mkdir(parents=True, exist_ok=True)
            output_file_path = current_output_dir / split_filename
            logger.info(f"Processing {split_filename} in '{dataset_name}' -> {output_file_path}")

            with open(llm_extracted_claims_path, "r", encoding="utf-8") as infile:
                all_claims_lines = infile.readlines()
            total_claims_to_process = len(all_claims_lines)
            if total_claims_to_process == 0: logger.info(f"No claims in {llm_extracted_claims_path}."); continue
            logger.info(f"Found {total_claims_to_process} claims in {split_filename}.")

            scores_from_fact_retrievals = []  # For P10 calculation

            with open(output_file_path, "w", encoding="utf-8") as outfile:
                for batch_start_idx in tqdm(range(0, total_claims_to_process, BATCH_SIZE),
                                            desc=f"Searching {split_filename} (queries) against {list(targeted_vector_store_dict.keys())} index(es)"):
                    batch_original_claims_data = []
                    for line_idx in range(batch_start_idx, min(batch_start_idx + BATCH_SIZE, total_claims_to_process)):
                        try:
                            batch_original_claims_data.append(json.loads(all_claims_lines[line_idx]))
                        except json.JSONDecodeError:
                            logger.warning(f"Malformed JSON in {llm_extracted_claims_path} at line ~{line_idx + 1}")

                    queries_for_batch_search = []
                    query_origin_info_list = []  # Stores (type, original_claim_idx_in_batch, original_claim_item, original_fact_item_if_any)

                    for claim_idx_in_batch, original_claim_item in enumerate(batch_original_claims_data):
                        facts_list = original_claim_item.get("facts", [])
                        processed_facts_for_claim = False
                        if facts_list and isinstance(facts_list, list):
                            for fact_item in facts_list:
                                query_str = fact_triplet_to_query_string(fact_item)
                                if query_str:
                                    queries_for_batch_search.append(query_str)
                                    query_origin_info_list.append(
                                        ("fact", claim_idx_in_batch, original_claim_item, fact_item))
                                    processed_facts_for_claim = True

                        if not processed_facts_for_claim:  # Fallback or if "facts" is empty/missing/not list
                            claim_text_query = clean_text(original_claim_item.get("claim", ""))
                            if claim_text_query:  # Ensure claim_text is not empty after cleaning
                                queries_for_batch_search.append(claim_text_query)
                                query_origin_info_list.append(
                                    ("claim_text", claim_idx_in_batch, original_claim_item, None))
                            else:  # If claim_text is also empty, log and prepare to skip search for this item
                                logger.debug(
                                    f"Empty query from both facts and claim_text for claim_id '{original_claim_item.get('claim_id', 'Unknown_ID_at_batch_idx_' + str(claim_idx_in_batch))}'.")
                                # Add a placeholder to keep query_origin_info_list aligned if needed, or handle carefully
                                # For now, search_faiss_indexes handles empty queries in its input list.

                    batch_search_results_map = {}  # Map: flat_query_idx -> {index_name_res: [(doc_obj, score), ...]}
                    if queries_for_batch_search:
                        batch_search_results_map = search_faiss_indexes(
                            targeted_vector_store_dict, queries_for_batch_search, TOP_K, targeted_max_distances_map)

                    # Aggregate results back to their original claim structure
                    results_grouped_by_original_claim_idx = defaultdict(
                        lambda: {"facts": [], "claim_text_retrieval": None})

                    for flat_query_idx, origin_info in enumerate(query_origin_info_list):
                        query_type, claim_idx_in_batch, _, original_fact_item = origin_info
                        retrieved_data_for_this_flat_query = batch_search_results_map.get(flat_query_idx,
                                                                                          {})  # This is {index_name_res: [(doc,score),...]}
                        query_string_used = queries_for_batch_search[flat_query_idx] if flat_query_idx < len(
                            queries_for_batch_search) else "QUERY_ERROR"

                        output_docs_for_current_query_across_indexes = {}  # {index_name: [filtered_docs_list]}
                        for index_name_res, found_docs_scores_tuples in retrieved_data_for_this_flat_query.items():
                            # --- Start of Duplicate Content Filtering for this (query, index) ---
                            filtered_docs_list_for_index = []
                            seen_contents_for_this_query_index = set()
                            for doc_obj, l2_score_val in found_docs_scores_tuples:
                                document_full_content = doc_obj.page_content
                                # Basic check for content existence, though page_content should ideally always be a string
                                if document_full_content is None:
                                    logger.warning(
                                        f"Document with None content encountered for query '{query_string_used[:50]}...' in index '{index_name_res}'. Skipping this doc.")
                                    continue

                                if document_full_content not in seen_contents_for_this_query_index:
                                    seen_contents_for_this_query_index.add(document_full_content)
                                    filtered_docs_list_for_index.append({
                                        "page_content": document_full_content,  # Storing full content
                                        "score": float(l2_score_val),
                                        "metadata": doc_obj.metadata
                                    })
                                    if query_type == "fact":  # Collect scores only for fact retrievals for P10
                                        scores_from_fact_retrievals.append(float(l2_score_val))
                                # else:
                                #     logger.debug(f"Skipped duplicate content for query '{query_string_used[:50]}...' in index '{index_name_res}'. Doc metadata: {doc_obj.metadata.get('source')}")
                            output_docs_for_current_query_across_indexes[index_name_res] = filtered_docs_list_for_index
                            # --- End of Duplicate Content Filtering ---

                        if query_type == "fact":
                            results_grouped_by_original_claim_idx[claim_idx_in_batch]["facts"].append(
                                {"original_fact": original_fact_item,
                                 "fact_query_string": query_string_used,
                                 "retrieved_relations": output_docs_for_current_query_across_indexes
                                 # Now contains filtered lists
                                 })
                        elif query_type == "claim_text":
                            results_grouped_by_original_claim_idx[claim_idx_in_batch]["claim_text_retrieval"] = {
                                "claim_text_query_string": query_string_used,
                                "retrieved_documents": output_docs_for_current_query_across_indexes
                                # Now contains filtered lists
                            }

                    # Write results for the current batch
                    for claim_idx_write, orig_claim_item_write in enumerate(batch_original_claims_data):
                        claim_id = orig_claim_item_write.get("claim_id")
                        ref_info = reference_data_map.get(claim_id, {})  # Safe get, defaults to {}

                        # Get aggregated data for this specific claim from the batch
                        aggregated_data_for_this_claim = results_grouped_by_original_claim_idx.get(claim_idx_write)
                        if not aggregated_data_for_this_claim:  # Should not happen if processing was correct
                            aggregated_data_for_this_claim = {"facts": [], "claim_text_retrieval": None}

                        output_record = {
                            "claim_id": claim_id,
                            "claim_text": orig_claim_item_write.get("claim", ""),
                            "label": ref_info.get("label", "UNKNOWN"),
                            "evidence_doc_ids": ref_info.get("evidence_doc_ids", []),
                            "retrieved_kg_relations_per_fact": aggregated_data_for_this_claim["facts"],
                            "retrieved_documents_for_claim_text": aggregated_data_for_this_claim["claim_text_retrieval"]
                        }
                        outfile.write(json.dumps(output_record) + "\n")

            # After processing all batches for a split_file
            logger.info(
                f"Analyzing fact retrieval scores for {split_filename} of {dataset_name} (targeted index(es): {list(targeted_vector_store_dict.keys())})...")
            # Calculate P10 based on all fact scores collected for this split file
            p10_threshold = analyze_scores(scores_from_fact_retrievals, split_filename, dataset_name,
                                           "_vs_".join(targeted_vector_store_dict.keys()))
            logger.info(
                f"P10 threshold for fact retrievals in {dataset_name}/{split_filename}: Score <= {p10_threshold:.4f} (L2 distance)")

            # Re-read, add strong_score, and re-write the file
            temp_lines_for_rewrite = []
            try:
                with open(output_file_path, "r", encoding="utf-8") as temp_infile:
                    for line in temp_infile:
                        try:
                            temp_lines_for_rewrite.append(json.loads(line))
                        except json.JSONDecodeError:
                            logger.warning(f"Malformed JSON during rewrite of {output_file_path}: {line.strip()}")
            except FileNotFoundError:
                logger.error(f"Output file {output_file_path} not found for rewrite. Skipping strong_score update.")
                continue  # Skip to next split_filename

            with open(output_file_path, "w", encoding="utf-8") as final_outfile:
                for record_data in temp_lines_for_rewrite:
                    # Add strong_score to relations retrieved from facts
                    if record_data.get("retrieved_kg_relations_per_fact"):
                        for fact_bundle in record_data["retrieved_kg_relations_per_fact"]:
                            if fact_bundle.get("retrieved_relations"):
                                for index_name_in_fact, relations_list in fact_bundle["retrieved_relations"].items():
                                    for relation_doc in relations_list:
                                        # L2 distance: lower is better, so score <= threshold is "strong"
                                        relation_doc["strong_score"] = bool(relation_doc["score"] <= p10_threshold)

                    # Optionally, add strong_score to documents retrieved for claim_text if a relevant threshold exists
                    # For now, only applying to fact retrievals as per original logic context
                    # if record_data.get("retrieved_documents_for_claim_text") and record_data["retrieved_documents_for_claim_text"].get("retrieved_documents"):
                    #     for index_name_claim_text, docs_list in record_data["retrieved_documents_for_claim_text"]["retrieved_documents"].items():
                    #         for doc_item in docs_list:
                    #             # You would need a separate threshold for claim_text retrievals if desired
                    #             # doc_item["strong_score"] = bool(doc_item["score"] <= some_other_threshold)
                    #             pass

                    final_outfile.write(json.dumps(record_data) + "\n")
            logger.info(
                f"Finished processing and updated {output_file_path} with strong_score flags for fact retrievals.")


if __name__ == "__main__":
    main()
    logger.info("Main processing completed (with duplicate content filtering per query per index).")