import os
import re
import shutil
from typing import List

import numpy as np
import torch
import faiss  # Ensure faiss is imported directly
import json
from tqdm import tqdm  # Ensure tqdm is imported
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.utils import DistanceStrategy

# ------------------ CONFIG ------------------
FAISS_INDEX_BASE_OUTPUT_DIR = "../../../data/Primary Evidance Source/_1_doc_as_kg_faiss_indexes_modernbert"

JSON_KG_DATABASES = [
    {
        "name": "Covid-Fact",
        "base_path": "../../../data/claims/kg_from_ref_docs/Covid-Fact",
        "json_files": ["kg_Covid-Fact_dev.json", "kg_Covid-Fact_test.json", "kg_Covid-Fact_train.json"],
        "output_dir_suffix": "Covid-Fact_triplets_flatL2",
        "max_items": None,
    },
    {
        "name": "HealthVer",
        "base_path": "../../../data/claims/kg_from_ref_docs/HealthVer",
        "json_files": ["kg_HealthVer_dev.json", "kg_HealthVer_test.json", "kg_HealthVer_train.json"],
        "output_dir_suffix": "HealthVer_triplets_flatL2",
        "max_items": None,
    },
    {
        "name": "SciFact",
        "base_path": "../../../data/claims/kg_from_ref_docs/SciFact",
        "json_files": ["kg_SciFact_corpus.json"],
        "output_dir_suffix": "SciFact_triplets_flatL2",
        "max_items": None,
    },
]

MODEL_NAME = "lightonai/modernbert-embed-large"
DEFAULT_MAX_ITEMS = 10000000
TRIPLET_UNKNOWN_PLACEHOLDER = "unknown"
EMBEDDING_BATCH_SIZE = 32  # Define a batch size for embedding


# ------------------ HELPERS (Largely Unchanged) ------------------

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r"[@#]\w+", "", text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else TRIPLET_UNKNOWN_PLACEHOLDER


def format_triplet_for_embedding(fact_data: dict) -> str:
    if not fact_data or not isinstance(fact_data, dict):
        return ""
    subj = clean_text(fact_data.get('source', TRIPLET_UNKNOWN_PLACEHOLDER))
    pred = clean_text(fact_data.get('relation', TRIPLET_UNKNOWN_PLACEHOLDER))
    obj = clean_text(fact_data.get('target', TRIPLET_UNKNOWN_PLACEHOLDER))
    if (subj == TRIPLET_UNKNOWN_PLACEHOLDER and
            pred == TRIPLET_UNKNOWN_PLACEHOLDER and
            obj == TRIPLET_UNKNOWN_PLACEHOLDER):
        return ""
    return f"{subj} [REL] {pred} [TAIL] {obj}"


def load_data_for_indexing(dataset_config):
    processed_items_map = {}
    doc_id_to_full_content = {}
    base_path = dataset_config["base_path"]
    kg_json_files = dataset_config.get("json_files", [])
    print(f"Processing dataset for indexing: {dataset_config['name']}")
    for kg_file_name in kg_json_files:
        kg_file_path = os.path.join(base_path, kg_file_name)
        if os.path.exists(kg_file_path):
            print(f"  Loading facts and source sentences from: {kg_file_path}")
            try:
                with open(kg_file_path, 'r', encoding='utf-8') as f:
                    kg_data_list = json.load(f)
                    if not isinstance(kg_data_list, list):
                        print(f"  ⚠️ KG file {kg_file_path} does not contain a list. Skipping.")
                        continue
                    for fact_item in kg_data_list:
                        doc_id = fact_item.get("document_id")
                        original_sentence_raw = fact_item.get("sentence")
                        full_content = fact_item.get("full_content")
                        if not (doc_id and original_sentence_raw): continue
                        if full_content: doc_id_to_full_content[doc_id] = full_content
                        cleaned_orig_sentence = clean_text(original_sentence_raw)
                        if not cleaned_orig_sentence or cleaned_orig_sentence == TRIPLET_UNKNOWN_PLACEHOLDER: continue
                        map_key = (doc_id, original_sentence_raw)
                        triplet_repr_text = format_triplet_for_embedding(fact_item)
                        text_to_embed_val, source_type_val = ("", "")
                        if triplet_repr_text:
                            text_to_embed_val, source_type_val = triplet_repr_text, "fact_derived_triplet"
                        else:
                            text_to_embed_val, source_type_val = cleaned_orig_sentence, "original_sentence_no_valid_triplet"
                            triplet_repr_text = None
                        processed_items_map[map_key] = {
                            "page_content_for_docstore": original_sentence_raw,
                            "text_to_embed": text_to_embed_val,
                            "metadata": {
                                "document_id": doc_id,
                                "original_sentence_raw": original_sentence_raw,
                                "full_content": doc_id_to_full_content.get(doc_id, "N/A"),
                                "source_type": source_type_val,
                                "triplet_representation_text_attempted": triplet_repr_text,
                                "fact_data": fact_item if triplet_repr_text else None,
                                "context": fact_item.get("context")}}
            except Exception as e:
                print(f"  ❌ Error reading KG file {kg_file_path}: {e}")
        else:
            print(f"  🤔 KG file not found: {kg_file_path}.")
    found_any_skipped_files = False
    for potential_skipped_file_name in os.listdir(base_path):
        if potential_skipped_file_name.startswith("skipped_sentences_") and potential_skipped_file_name.endswith(
                ".json"):
            found_any_skipped_files = True
            skipped_file_path = os.path.join(base_path, potential_skipped_file_name)
            print(f"  Loading explicitly skipped sentences from: {skipped_file_path}")
            try:
                with open(skipped_file_path, 'r', encoding='utf-8') as f:
                    skipped_data_dict = json.load(f)
                    if not isinstance(skipped_data_dict, dict):
                        print(f"  ⚠️ Skipped sentences file {skipped_file_path} is not a dictionary. Skipping.")
                        continue
                    for doc_id, sentences_list in skipped_data_dict.items():
                        current_doc_full_content = doc_id_to_full_content.get(doc_id, "N/A")
                        if not isinstance(sentences_list, list): continue
                        for skipped_sentence_raw in sentences_list:
                            if not skipped_sentence_raw: continue
                            map_key = (doc_id, skipped_sentence_raw)
                            if map_key not in processed_items_map:
                                cleaned_skipped_sentence = clean_text(skipped_sentence_raw)
                                if not cleaned_skipped_sentence or cleaned_skipped_sentence == TRIPLET_UNKNOWN_PLACEHOLDER: continue
                                processed_items_map[map_key] = {
                                    "page_content_for_docstore": skipped_sentence_raw,
                                    "text_to_embed": cleaned_skipped_sentence,
                                    "metadata": {
                                        "document_id": doc_id,
                                        "original_sentence_raw": skipped_sentence_raw,
                                        "full_content": current_doc_full_content,
                                        "source_type": "explicitly_skipped_sentence_text",
                                        "triplet_representation_text_attempted": None,
                                        "fact_data": None, "context": None}}
            except Exception as e:
                print(f"  ❌ Error reading skipped sentences file {skipped_file_path}: {e}")
    if not found_any_skipped_files: print(f"  🤔 No 'skipped_sentences_*.json' files found in {base_path}.")
    final_items_list = list(processed_items_map.values())
    print(f"  Total unique items prepared for {dataset_config['name']}: {len(final_items_list)}")
    return final_items_list


def build_faiss_flat_index(docs: List[Document], embeddings_model, start_index=0, batch_size=32):  # Added batch_size
    if not docs:
        print("    ⚠️ build_faiss_flat_index called with no documents. Returning None.")
        return None

    texts_to_embed = []
    for doc in docs:
        text_val = doc.metadata.get("text_to_embed_final")
        if text_val:
            texts_to_embed.append(text_val)
        else:
            print(f"    Warning: 'text_to_embed_final' missing. Using page_content. Doc: {doc.page_content[:50]}")
            texts_to_embed.append(doc.page_content)

    if not texts_to_embed:
        print("    ⚠️ No texts extracted from documents to embed. Returning None.")
        return None

    all_embeddings_list = []
    # Process texts in batches for embedding with tqdm progress bar
    # print(f"    Embedding {len(texts_to_embed)} texts (triplets or sentences) in batches of {batch_size}...")
    for i in tqdm(range(0, len(texts_to_embed), batch_size), desc="    Embedding texts in batches"):
        batch_texts = texts_to_embed[i:i + batch_size]
        batch_embeddings = embeddings_model.embed_documents(batch_texts)
        all_embeddings_list.extend(batch_embeddings)

    embeddings_np = np.array(all_embeddings_list).astype("float32")
    print(f"    Embeddings created with shape: {embeddings_np.shape}")

    if embeddings_np.shape[0] == 0:
        print("    No embeddings to process. Skipping FAISS index creation.")
        return None

    dimension = embeddings_np.shape[1]
    print(f"    Normalizing {embeddings_np.shape[0]} embeddings L2.")
    faiss.normalize_L2(embeddings_np)
    print(f"    Creating IndexFlatL2 with dimension: {dimension}")
    final_faiss_index = faiss.IndexFlatL2(dimension)
    print(f"    Adding embeddings to IndexFlatL2...")
    final_faiss_index.add(embeddings_np)
    print(f"    Total vectors in index: {final_faiss_index.ntotal}")

    doc_ids_map = [str(i + start_index) for i in range(len(docs))]
    docstore = InMemoryDocstore({doc_id: doc for doc_id, doc in zip(doc_ids_map, docs)})
    index_to_docstore_id = {i: doc_id for i, doc_id in enumerate(doc_ids_map)}

    return FAISS(
        embedding_function=embeddings_model.embed_query,
        index=final_faiss_index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
        normalize_L2=False,
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE)


# ------------------ MAIN ------------------

print("Initializing embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={"device": 0 if torch.cuda.is_available() else -1, "trust_remote_code": True},
    encode_kwargs={"normalize_embeddings": True},
    cache_folder="../../models/embeddings/")

for kg_config in JSON_KG_DATABASES:
    dataset_name = kg_config["name"]
    print(f"\n🚀 Processing Dataset: {dataset_name} (indexing triplets AND skipped sentences with IndexFlatL2)")
    OUTPUT_DIR = os.path.join(FAISS_INDEX_BASE_OUTPUT_DIR, kg_config["output_dir_suffix"])
    MAX_ITEMS_TO_PROCESS = kg_config.get("max_items") or DEFAULT_MAX_ITEMS
    if os.path.exists(OUTPUT_DIR):
        print(f"⚠️ Output directory {OUTPUT_DIR} already exists. Removing old index files.")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📥 Loading and preparing data for indexing: {dataset_name}...")
    doc_input_data_list = load_data_for_indexing(kg_config)
    if not doc_input_data_list:
        print(f"🤷 No data found for dataset {dataset_name}. Skipping index creation.")
        continue
    print(f"🔎 Found {len(doc_input_data_list)} items to process for {dataset_name}.")
    if MAX_ITEMS_TO_PROCESS and len(doc_input_data_list) > MAX_ITEMS_TO_PROCESS:
        print(f"✂️ Truncating item list for {dataset_name} to {MAX_ITEMS_TO_PROCESS}.")
        doc_input_data_list = doc_input_data_list[:MAX_ITEMS_TO_PROCESS]
    all_langchain_documents = []
    print(f"📄 Creating Langchain Document objects for {len(doc_input_data_list)} items from {dataset_name}...")
    for item_data in tqdm(doc_input_data_list, desc=f"Creating Documents for {dataset_name}"):
        current_metadata = item_data["metadata"]
        current_metadata["text_to_embed_final"] = item_data["text_to_embed"]
        doc = Document(page_content=item_data["page_content_for_docstore"], metadata=current_metadata)
        all_langchain_documents.append(doc)
    if not all_langchain_documents:
        print(f"🤷 No Langchain documents created. Skipping FAISS index creation.")
        continue
    print(f"\n💾 Building and saving FAISS IndexFlatL2 for {len(all_langchain_documents)} items from {dataset_name}...")
    final_index_wrapper = build_faiss_flat_index(
        all_langchain_documents,
        embedding_model,
        start_index=0,
        batch_size=EMBEDDING_BATCH_SIZE  # Pass batch size
    )
    if final_index_wrapper:
        final_index_wrapper.save_local(OUTPUT_DIR)
        print(f"✅ Final FAISS index for {dataset_name} saved to: {OUTPUT_DIR}")
    else:
        print(f"❌ Failed to build FAISS index for {dataset_name}.")
print("\n🎉 All datasets processed.")