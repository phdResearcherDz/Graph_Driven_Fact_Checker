import json
import os
import time
import re
from typing import List, Dict, Set, Optional
from pydantic import BaseModel, ValidationError, constr

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama.llms import OllamaLLM


# =======================
# 1. SCHEMA
# =======================
class MedicalFact(BaseModel):
    subject: constr(strip_whitespace=True, min_length=1)
    predicate: constr(strip_whitespace=True, min_length=1)
    object: constr(strip_whitespace=True, min_length=1)


class FactExtractionOutput(BaseModel):
    facts: List[MedicalFact]


# =======================
# 2. LLM & PROMPT
# =======================
llm = OllamaLLM(model="gemma3:12b")  # Replace with your local model

extraction_prompt_template = PromptTemplate(
    input_variables=["sentence", "full_text", "previous_facts", "format_instructions"],
    template="""
You are a medical fact extractor AI assistant. Your task is to extract **semantic** and **medically meaningful** facts from a given sentence in a biomedical passage. Facts should be expressed as subject-predicate-object triplets.

### Semantic Rules:
- Extract the core medical meaning, not just a surface phrase.
- Choose the most **medically relevant** and **standardized** term for each component.
- Rephrase or normalize the extracted terms for clarity (e.g., "helps reduce pain" → "reduces pain").

### Do not:
- Copy text verbatim unless it's already a clean fact.
- Include vague predicates like "helps", "causes problems", or "can be used".
- Leave any pronouns (it, they, this) in the triplet.
- Extract non-medical or ambiguous general knowledge.

### Sentence to Process:
{sentence}

### Previously Extracted Medical Facts (as JSON array of objects, e.g., [{{ "subject": "...", "predicate": "...", "object": "..." }}]):
{previous_facts}

{format_instructions}
"""
)

extraction_parser = JsonOutputParser(pydantic_object=FactExtractionOutput)
extraction_chain = extraction_prompt_template | llm | extraction_parser


# =======================
# 3. HELPERS
# =======================
def split_into_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def normalize_fact(fact: Dict[str, str]) -> str:
    subject = fact.get("subject", "")
    predicate = fact.get("predicate", "")
    object_ = fact.get("object", "")
    return " | ".join([
        normalize_text(subject),
        normalize_text(predicate),
        normalize_text(object_)
    ])


# =======================
# 4. CORE EXTRACTION FUNCTION
# =======================
def extract_facts_from_sentence(sentence: str, full_text: str, previous_facts: List[Dict], retries: int = 5) -> List[
    Dict]:
    attempt = 0
    format_instructions = extraction_parser.get_format_instructions()
    while attempt < retries:
        try:
            result = extraction_chain.invoke({
                "sentence": sentence,
                "full_text": full_text,
                "previous_facts": json.dumps(previous_facts, indent=2),
                "format_instructions": format_instructions
            })
            extracted_facts = result.get("facts", [])
            if not isinstance(extracted_facts, list):
                print(f"⚠️ LLM returned non-list for facts. Retrying.")
                attempt += 1
                time.sleep(0.5)
                continue

            validated_facts = []
            for fact_data in extracted_facts:
                try:
                    validated_fact = MedicalFact(**fact_data)
                    validated_facts.append(validated_fact.model_dump())
                except ValidationError as e:
                    print(f"⚠️ Invalid fact structure in LLM output: {fact_data}. Error: {e}")
                except TypeError as e:
                    print(f"⚠️ Type error with fact data: {fact_data}. Error: {e}")
            return [{"fact": f, "sentence": sentence} for f in validated_facts]
        except (ValidationError, Exception) as e:
            print(f"⚠️ Retry {attempt + 1}: {e}")
            attempt += 1
            time.sleep(0.5)
    print(f"❌ Failed to extract facts after {retries} attempts for sentence: {sentence[:60]}...")
    return []


# =======================
# 5. MAIN FUNCTION FOR FACT EXTRACTION
# =======================
def extract_from_text(text: str) -> Dict[str, List[Dict]]:
    sentences = split_into_sentences(text)
    all_facts_with_sentences = []
    seen: Set[str] = set()

    for i, sentence in enumerate(sentences):
        # Print progression for each sentence
        print(f"\n🔍 Processing sentence {i + 1}/{len(sentences)}: {sentence[:90]}{'...' if len(sentence) > 90 else ''}")
        new_facts_with_sentence = extract_facts_from_sentence(
            sentence,
            text,
            [f["fact"] for f in all_facts_with_sentences if "fact" in f]
        )

        for item in new_facts_with_sentence:
            fact = item.get("fact")
            original_sentence = item.get("sentence")

            if fact and original_sentence:
                key = normalize_fact(fact)
                if key not in seen:
                    seen.add(key)
                    all_facts_with_sentences.append({"fact": fact, "sentence": original_sentence})
                    print(
                        f"  ✅ New fact added: {key} (from: {original_sentence[:60]}{'...' if len(original_sentence) > 60 else ''})")
                else:
                    print(f"  🔁 Skipped duplicate: {key}")
            else:
                print(f"  ⚠️ Skipped malformed item from extraction: {item}")

    print(f"\n📦 Total unique facts extracted from this text: {len(all_facts_with_sentences)}")
    return {"facts_with_sentence": all_facts_with_sentences}


# =======================
# 6. FUNCTION TO BUILD KNOWLEDGE GRAPH FROM FILE
# =======================
def build_knowledge_graph_from_file(file_path: str) -> List[Dict]:
    """
    Loads documents from a specified file path (each line should be a JSON object)
    and builds a knowledge graph using LLM-based fact extraction for all documents at once.

    Args:
        file_path (str): The path to the file containing the JSON documents.
    """
    knowledge_graph = []
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return knowledge_graph

    all_docs_content = {}
    doc_count = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                doc = json.loads(line.strip())
                if not isinstance(doc, dict) or "id" not in doc or "title" not in doc or "content" not in doc:
                    print(f"Warning: Skipping invalid JSON object on line {line_num}: {line.strip()[:100]}...")
                    continue
                doc_id = doc["id"]
                content = doc.get("content", "")
                all_docs_content[doc_id] = content
                doc_count += 1
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line {line_num}: {line.strip()[:100]}...")
            except Exception as e:
                print(f"An error occurred while processing line {line_num}: {e}")

    print(f"\n📝 Loaded {doc_count} documents from {file_path}. Starting fact extraction...")

    processed_docs = 0
    total_docs = len(all_docs_content)

    for doc_id, content in all_docs_content.items():
        processed_docs += 1
        print(f"\n--- Document {processed_docs}/{total_docs} (ID: {doc_id}) ---")
        print(f"🚀 Extracting facts from document '{doc_id}' (length: {len(content)} chars)")

        result = extract_from_text(content)
        content_facts_with_sentence = result.get("facts_with_sentence", [])

        extracted_facts_count = 0
        for item in content_facts_with_sentence:
            fact = item.get("fact")
            original_sentence = item.get("sentence")

            if fact and original_sentence:
                knowledge_graph.append({
                    "source": fact.get("subject", ""),
                    "relation": fact.get("predicate", ""),
                    "target": fact.get("object", ""),
                    "document_id": doc_id,
                    "context": "content",
                    "sentence": original_sentence,
                    "full_content": content
                })
                extracted_facts_count += 1
            else:
                print(f"⚠️ Skipping malformed fact/sentence for document {doc_id}: {item}")
        print(f"✨ Finished document '{doc_id}'. Extracted {extracted_facts_count} unique facts from this document.")

    return knowledge_graph


# =======================
# 7. MAIN EXECUTION
# =======================
if __name__ == "__main__":
    input_root = "../../data/claims/processed"
    output_root = "../../data/kg_from_ref_docs_llm_all_at_once"
    os.makedirs(output_root, exist_ok=True)

    skipped_datasets = ["CoVERT", "Fever","HealthVer","Covid-Fact","PubHealth","VitaminC"]

    total_files_processed = 0
    total_files_skipped = 0

    # Get a list of all files to process to provide overall progress
    files_to_process = []
    for dataset_name in os.listdir(input_root):
        dataset_path = os.path.join(input_root, dataset_name)
        if not os.path.isdir(dataset_path) or dataset_name in skipped_datasets:
            continue
        docs_path = os.path.join(dataset_path, "docs")
        if not os.path.exists(docs_path):
            continue
        for file_name in os.listdir(docs_path):
            if file_name.endswith(".jsonl"):
                files_to_process.append(os.path.join(docs_path, file_name))

    total_jsonl_files = len(files_to_process)
    current_file_idx = 0

    print(f"\n--- Starting Knowledge Graph Construction ---")
    print(f"Total .jsonl files to process: {total_jsonl_files}")

    for input_file_path in files_to_process:
        current_file_idx += 1
        dataset_name = os.path.basename(os.path.dirname(os.path.dirname(input_file_path)))
        file_name = os.path.basename(input_file_path)

        print(
            f"\n===== Processing File {current_file_idx}/{total_jsonl_files}: {file_name} (Dataset: {dataset_name}) =====")

        dataset_output_folder = os.path.join(output_root, dataset_name)
        os.makedirs(dataset_output_folder, exist_ok=True)

        try:
            kg = build_knowledge_graph_from_file(input_file_path)
            output_file = os.path.join(dataset_output_folder,
                                       f"kg_{dataset_name}_{file_name.replace('.jsonl', '.json')}")
            with open(output_file, 'w', encoding='utf-8') as outfile:
                json.dump(kg, outfile, indent=2)
            print(f"🟢 Knowledge graph for {file_name} saved to: {output_file} with {len(kg)} facts.")
            total_files_processed += 1
        except Exception as e:
            print(f"🔴 Error processing {file_name}: {e}")
            total_files_skipped += 1

        print(f"===== Finished File {current_file_idx}/{total_jsonl_files}: {file_name} =====\n")

    print(f"\n--- Knowledge Graph Construction Complete ---")
    print(f"Summary:")
    print(f"  Processed {total_files_processed} files successfully.")
    print(f"  Skipped {total_files_skipped} files due to errors.")
    print(f"  Total files considered: {total_jsonl_files}.")