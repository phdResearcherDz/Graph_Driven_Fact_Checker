import json
from pathlib import Path
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import logging
import re  # For parsing filenames

# ------------------ CONFIG ------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# INPUT_ROOT points to the output of your search script
INPUT_ROOT = Path("../../../data/_3_Results/docs_kg_structure_filtered")
# EVAL_OUTPUT_ROOT is where this evaluation script will save its results
EVAL_OUTPUT_ROOT = Path("./docs_kg_structure_evaluation_filtered")

K_VALUES = [1, 3, 5, 10, 20, 30, 40, 50]  # User updated K_VALUES
MAX_RANK_FOR_HISTOGRAM = K_VALUES[-1]

LABEL_SUPPORTED = "SUPPORTED"
LABEL_REFUTED = "REFUTED"
LABEL_NEI = "NOT ENOUGH INFO"
LABEL_UNKNOWN = "UNKNOWN"
ORDERED_LABELS_FOR_PLOTTING = [LABEL_SUPPORTED, LABEL_REFUTED, LABEL_NEI, LABEL_UNKNOWN]


# ------------------ METRIC CALCULATION HELPERS (Unchanged) ------------------
def precision_at_k(retrieved_k_docs, ground_truth_docs_set, k_val):
    if k_val == 0: return 0.0
    if not retrieved_k_docs: return 0.0
    relevant_found = sum(1 for doc_id in retrieved_k_docs if doc_id in ground_truth_docs_set)
    return relevant_found / k_val


def recall_at_k(retrieved_k_docs, ground_truth_docs_set, num_total_relevant_docs):
    if num_total_relevant_docs == 0: return 0.0
    if not retrieved_k_docs: return 0.0
    relevant_found = sum(1 for doc_id in retrieved_k_docs if doc_id in ground_truth_docs_set)
    return relevant_found / num_total_relevant_docs


def f1_score_at_k(p_k, r_k):
    if p_k + r_k == 0: return 0.0
    return 2 * (p_k * r_k) / (p_k + r_k)


def get_reciprocal_rank_and_first_rank(ordered_retrieved_docs, ground_truth_docs_set):
    if not ordered_retrieved_docs: return 0.0, float('inf')
    for rank_idx, doc_id in enumerate(ordered_retrieved_docs):
        if doc_id in ground_truth_docs_set:
            first_rank = rank_idx + 1
            return 1.0 / first_rank, first_rank
    return 0.0, float('inf')


# ------------------ NEW METRICS REPORTING HELPER ------------------
def _calculate_and_log_metrics(metrics_data: dict, k_values: list, num_claims: int, logger) -> dict:
    """Calculates, logs, and returns a summary dict for a given set of metric values."""
    summary_dict = {}
    if not num_claims:
        logger.info("  No claims with ground truth in this group. Skipping metrics.")
        return summary_dict

    # --- MRR ---
    valid_rrs = [r for r in metrics_data.get("rr", []) if r is not None and r != float('inf')]
    if valid_rrs:
        mrr = np.mean(valid_rrs)
        summary_dict["MRR"] = mrr
        logger.info(f"  MRR: {mrr:.4f} (over {len(valid_rrs)} claims)")
    else:
        summary_dict["MRR"] = 0.0
        logger.info(f"  MRR: 0.0 (no relevant documents found)")

    # --- P, R, F1 @ K ---
    for k in k_values:
        for metric_prefix in ["precision", "recall", "f1"]:
            metric_key = f"{metric_prefix}_at_{k}"
            display_key = f"Mean_{metric_prefix.capitalize()}@{k}"
            values = [v for v in metrics_data.get(metric_key, []) if v is not None and not np.isnan(v)]
            if values:
                mean_val = np.mean(values)
                summary_dict[display_key] = float(mean_val)
                logger.info(f"  {display_key}: {float(mean_val):.4f} (over {len(values)} claims)")
            else:
                summary_dict[display_key] = 0.0
                logger.info(f"  {display_key}: 0.0 (no valid data)")
    return summary_dict


# ------------------ LABEL NORMALIZATION HELPER (Unchanged) ------------------
def normalize_label(raw_label: str) -> str:
    if not raw_label: return LABEL_UNKNOWN
    label_upper = raw_label.strip().upper()
    if label_upper == "SUPPORTS" or label_upper == LABEL_SUPPORTED:
        return LABEL_SUPPORTED
    elif label_upper == "REFUTES" or label_upper == LABEL_REFUTED:
        return LABEL_REFUTED
    elif "NOT ENOUGH" in label_upper or "NEI" in label_upper:
        return LABEL_NEI
    else:
        return LABEL_UNKNOWN


# ------------------ FILENAME PARSING HELPER (REVISED) ------------------
def parse_search_output_filename(filename_str: str):
    # Expected format now: dev.jsonl, test.jsonl, or train.jsonl
    match = re.match(r"(dev|test|train)\.jsonl", filename_str, re.IGNORECASE)
    if match:
        file_prefix = match.group(1).lower()  # dev, test, or train
        return file_prefix
    logger.warning(f"Could not parse file prefix (dev/test/train) from filename: {filename_str}")
    return None


# ------------------ MAIN EVALUATION LOGIC (REVISED FOR PER-CLASS METRICS) ------------------
def evaluate_retrieval(input_root: Path, output_root: Path, k_values: list):
    if not input_root.exists():
        logger.error(f"Input directory {input_root} does not exist. Exiting.")
        return
    output_root.mkdir(parents=True, exist_ok=True)

    # MODIFIED: This accumulator now has a level for the claim label.
    # Structure: { index_name: { label: { metric_name: [values] } } }
    per_label_metrics_accumulator = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    first_relevant_ranks_by_file_label = defaultdict(lambda: defaultdict(list))
    hit_at_k_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'hits': 0, 'total': 0})))
    detailed_results_output_path = output_root / "evaluation_details_per_claim.jsonl"
    processed_files_count = 0

    with open(detailed_results_output_path, 'w', encoding='utf-8') as detailed_outfile:
        # Iterate through dataset subdirectories in INPUT_ROOT
        for dataset_dir in input_root.iterdir():
            if not dataset_dir.is_dir():
                continue

            index_name_for_metrics = dataset_dir.name  # e.g., "Covid-Fact"

            # Look for dev.jsonl, test.jsonl, train.jsonl directly
            for expected_file_prefix in ["dev", "test", "train"]:
                file_path = dataset_dir / f"{expected_file_prefix}.jsonl"

                if not file_path.exists():
                    continue

                filename = file_path.name
                file_prefix_part = parse_search_output_filename(filename)
                if not file_prefix_part:
                    logger.error(f"Logic error: Could not parse known filename pattern: {file_path}")
                    continue

                file_id_for_plots = f"{index_name_for_metrics}_{file_prefix_part}"

                if index_name_for_metrics.lower() == "scifact" and "test" == file_prefix_part.lower():
                    logger.info(f"Skipping SciFact test file: {file_path} (standard skip for this dataset/split)")
                    continue

                processed_files_count += 1
                logger.info(
                    f"Processing file: {file_path} (Index: {index_name_for_metrics}, File ID for plots: {file_id_for_plots})")

                records_to_process = []
                try:
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        for line_num, line in enumerate(infile, 1):
                            try:
                                record = json.loads(line)
                                records_to_process.append(record)
                            except json.JSONDecodeError:
                                logger.error(
                                    f"Malformed JSON on line {line_num} in {file_path}. Skipping line: {line.strip()}")
                                continue
                except Exception as e:
                    logger.error(f"Error reading file {file_path}: {e}. Skipping file.")
                    continue

                if not records_to_process:
                    logger.warning(f"No valid JSON records found or read from {file_path}. Skipping.")
                    continue

                for record_idx, record in enumerate(records_to_process):
                    try:
                        claim_id = record.get("claim_id", f"unknown_claim_{file_id_for_plots}_record_{record_idx}")
                        claim_text = record.get("claim_text", "")
                        raw_claim_label = record.get("label", "UNKNOWN")
                        normalized_claim_label = normalize_label(raw_claim_label)
                        ground_truth_doc_ids_list = record.get("evidence_doc_ids", [])
                        if not isinstance(ground_truth_doc_ids_list, list): ground_truth_doc_ids_list = []
                        ground_truth_doc_ids_set = set(filter(None, ground_truth_doc_ids_list))
                        num_total_relevant_for_query = len(ground_truth_doc_ids_set)

                        claim_detail_base = {
                            "file_id_for_plots": file_id_for_plots,
                            "index_name_for_metrics": index_name_for_metrics,
                            "claim_id": claim_id, "claim_text": claim_text,
                            "label": normalized_claim_label,
                            "ground_truth_doc_ids": list(ground_truth_doc_ids_set),
                            "num_ground_truth": num_total_relevant_for_query,
                        }

                        if num_total_relevant_for_query == 0:
                            logger.debug(
                                f"Claim {claim_id} in {file_id_for_plots} has no ground truth. Skipping metrics.")
                            claim_detail_base["evaluation_skipped_reason"] = "No ground truth evidence"
                            claim_detail_base["retrieved_docs_info_by_index"] = None
                            detailed_outfile.write(json.dumps(claim_detail_base) + "\n")
                            continue

                        aggregated_doc_scores_by_index = defaultdict(list)
                        fact_bundles = record.get("retrieved_kg_relations_per_fact", [])
                        claim_text_retrieval_bundle = record.get("retrieved_documents_for_claim_text", None)

                        if fact_bundles and isinstance(fact_bundles, list) and any(
                                b.get("retrieved_relations") for b in fact_bundles):
                            for bundle in fact_bundles:
                                retrieved_relations_dict = bundle.get("retrieved_relations", {})
                                for src_index_name, relations_list_for_fact in retrieved_relations_dict.items():
                                    for rel_data in relations_list_for_fact:
                                        metadata = rel_data.get("metadata", {})
                                        doc_id = metadata.get("document_id")
                                        score = rel_data.get("score")
                                        if doc_id and score is not None:
                                            aggregated_doc_scores_by_index[src_index_name].append((doc_id, score))
                        elif claim_text_retrieval_bundle and isinstance(claim_text_retrieval_bundle,
                                                                        dict) and claim_text_retrieval_bundle.get(
                            "retrieved_documents"):
                            retrieved_documents_dict = claim_text_retrieval_bundle.get("retrieved_documents", {})
                            for src_index_name, documents_list in retrieved_documents_dict.items():
                                for doc_data in documents_list:
                                    metadata = doc_data.get("metadata", {})
                                    doc_id = metadata.get("document_id")
                                    score = doc_data.get("score")
                                    if doc_id and score is not None:
                                        aggregated_doc_scores_by_index[src_index_name].append((doc_id, score))

                        target_index_name = index_name_for_metrics
                        claim_detail_retrieved_docs_info = {}
                        doc_score_pairs_for_target_index = aggregated_doc_scores_by_index.get(target_index_name, [])

                        if not doc_score_pairs_for_target_index:
                            rr_value, first_rank = 0.0, float('inf')
                            # MODIFICATION: Add metrics to the per-label accumulator
                            per_label_metrics_accumulator[target_index_name][normalized_claim_label]["rr"].append(
                                rr_value)
                            first_relevant_ranks_by_file_label[file_id_for_plots][normalized_claim_label].append(
                                first_rank)
                            claim_detail_retrieved_docs_info[target_index_name] = {"ordered_retrieved_doc_ids": [],
                                                                                   "first_relevant_rank": float('inf')}
                            for k_val in k_values:
                                # MODIFICATION: Add metrics to the per-label accumulator
                                per_label_metrics_accumulator[target_index_name][normalized_claim_label][
                                    f"precision_at_{k_val}"].append(0.0)
                                per_label_metrics_accumulator[target_index_name][normalized_claim_label][
                                    f"recall_at_{k_val}"].append(0.0)
                                per_label_metrics_accumulator[target_index_name][normalized_claim_label][
                                    f"f1_at_{k_val}"].append(0.0)
                                hit_at_k_counts[file_id_for_plots][normalized_claim_label][k_val]['total'] += 1
                                claim_detail_retrieved_docs_info[target_index_name][f"hit_at_{k_val}"] = False
                        else:
                            sorted_unique_docs = {}
                            for doc_id, score in sorted(doc_score_pairs_for_target_index,
                                                        key=lambda x: x[1]):  # Scores are L2 distance, lower is better
                                if doc_id not in sorted_unique_docs:
                                    sorted_unique_docs[doc_id] = score
                            ordered_retrieved_doc_ids_for_claim = [p[0] for p in sorted(sorted_unique_docs.items(),
                                                                                        key=lambda x: x[1])]
                            rr_value, first_rank = get_reciprocal_rank_and_first_rank(
                                ordered_retrieved_doc_ids_for_claim, ground_truth_doc_ids_set)

                            # MODIFICATION: Add metrics to the per-label accumulator
                            per_label_metrics_accumulator[target_index_name][normalized_claim_label]["rr"].append(
                                rr_value)
                            first_relevant_ranks_by_file_label[file_id_for_plots][normalized_claim_label].append(
                                first_rank)

                            current_claim_retrieved_info_for_index = {
                                "ordered_retrieved_doc_ids": ordered_retrieved_doc_ids_for_claim[:max(k_values)],
                                "first_relevant_rank": first_rank if first_rank != float('inf') else None,
                            }
                            for k_val in k_values:
                                retrieved_top_k_docs = ordered_retrieved_doc_ids_for_claim[:k_val]
                                p_k = precision_at_k(retrieved_top_k_docs, ground_truth_doc_ids_set, k_val)
                                r_k = recall_at_k(retrieved_top_k_docs, ground_truth_doc_ids_set,
                                                  num_total_relevant_for_query)
                                f1_k = f1_score_at_k(p_k, r_k)

                                # MODIFICATION: Add metrics to the per-label accumulator
                                per_label_metrics_accumulator[target_index_name][normalized_claim_label][
                                    f"precision_at_{k_val}"].append(p_k)
                                per_label_metrics_accumulator[target_index_name][normalized_claim_label][
                                    f"recall_at_{k_val}"].append(r_k)
                                per_label_metrics_accumulator[target_index_name][normalized_claim_label][
                                    f"f1_at_{k_val}"].append(f1_k)

                                hit_at_k_counts[file_id_for_plots][normalized_claim_label][k_val]['total'] += 1
                                is_hit = any(doc_id in ground_truth_doc_ids_set for doc_id in retrieved_top_k_docs)
                                if is_hit: hit_at_k_counts[file_id_for_plots][normalized_claim_label][k_val][
                                    'hits'] += 1
                                current_claim_retrieved_info_for_index[f"hit_at_{k_val}"] = is_hit
                            claim_detail_retrieved_docs_info[target_index_name] = current_claim_retrieved_info_for_index

                        claim_detail_output = {**claim_detail_base,
                                               "retrieved_docs_info_by_index": claim_detail_retrieved_docs_info}
                        detailed_outfile.write(json.dumps(claim_detail_output) + "\n")
                    except Exception as e:
                        logger.error(
                            f"Error processing record {record_idx} in {file_path} for claim_id '{record.get('claim_id', 'N/A')}': {e}",
                            exc_info=True)

    # --- START OF COMPLETELY REVISED REPORTING SECTION ---

    if processed_files_count == 0:
        logger.warning(
            f"No '.jsonl' files processed in {input_root} matching expected pattern. No evaluation performed.")
        return

    final_metrics_summary = {}
    logger.info("\n" + "=" * 80)
    logger.info("--- Evaluation Results (Claim-Level Retrieval - Per-Label and Overall) ---")
    logger.info("=" * 80)

    for index_name, per_label_data in per_label_metrics_accumulator.items():
        logger.info(f"\n\nMetrics for Index/Dataset: '{index_name}'")
        logger.info("-" * 50)

        final_metrics_summary[index_name] = {}
        overall_metrics_aggregator = defaultdict(list)
        total_claims_with_gt_for_index = 0

        # Calculate and log metrics for each label individually
        sorted_labels = sorted(per_label_data.keys(), key=lambda x: ORDERED_LABELS_FOR_PLOTTING.index(
            x) if x in ORDERED_LABELS_FOR_PLOTTING else 99)

        for label in sorted_labels:
            metrics_for_label = per_label_data[label]
            num_claims_for_label = len(metrics_for_label.get("rr", []))
            total_claims_with_gt_for_index += num_claims_for_label

            logger.info(f"\n--- Metrics for Label: '{label}' ({num_claims_for_label} claims with ground truth) ---")

            # Use the new helper to calculate metrics for the current label
            label_summary = _calculate_and_log_metrics(metrics_for_label, k_values, num_claims_for_label, logger)
            final_metrics_summary[index_name][label] = label_summary

            # Aggregate all raw values to calculate the overall metrics later
            for metric, values in metrics_for_label.items():
                overall_metrics_aggregator[metric].extend(values)

        # Calculate and log the OVERALL metrics for the entire index
        logger.info(
            f"\n--- Overall Metrics for '{index_name}' ({total_claims_with_gt_for_index} total claims with ground truth) ---")
        overall_summary = _calculate_and_log_metrics(overall_metrics_aggregator, k_values,
                                                     total_claims_with_gt_for_index, logger)
        final_metrics_summary[index_name]["Overall"] = overall_summary

    # --- END OF COMPLETELY REVISED REPORTING SECTION ---

    metrics_output_path = output_root / "evaluation_summary_metrics_claim_level.json"
    with open(metrics_output_path, 'w', encoding='utf-8') as f:
        json.dump(final_metrics_summary, f, indent=4)
    logger.info(f"\n\nSummary metrics saved to: {metrics_output_path}")
    logger.info(f"Detailed per-claim evaluation results saved to: {detailed_results_output_path}")

    logger.info("\n--- Generating Detailed Rank Histograms (per file_id, per label) ---")
    label_colors = {LABEL_SUPPORTED: "green", LABEL_REFUTED: "red", LABEL_NEI: "blue", LABEL_UNKNOWN: "grey"}
    for file_id, ranks_data_by_label in first_relevant_ranks_by_file_label.items():
        plt.figure(figsize=(14, 8));
        total_queries_in_file_with_gt_and_rank = 0;
        has_plotted_data_for_file = False
        for label_key in ORDERED_LABELS_FOR_PLOTTING:
            ranks = ranks_data_by_label.get(label_key, [])
            if not ranks: continue
            total_queries_in_file_with_gt_and_rank += len(ranks)
            capped_ranks = [min(r, MAX_RANK_FOR_HISTOGRAM + 1) for r in ranks]
            not_found_or_beyond_max_for_label = sum(1 for r in capped_ranks if r > MAX_RANK_FOR_HISTOGRAM)
            plot_ranks_for_label = [r for r in capped_ranks if r <= MAX_RANK_FOR_HISTOGRAM]
            hist_legend_label = f"{label_key} (n={len(ranks)})"
            if not_found_or_beyond_max_for_label > 0: hist_legend_label += f"; >{MAX_RANK_FOR_HISTOGRAM}/NF: {not_found_or_beyond_max_for_label}"
            if plot_ranks_for_label:
                bins = np.arange(1, MAX_RANK_FOR_HISTOGRAM + 2) - 0.5
                plt.hist(plot_ranks_for_label, bins=bins, rwidth=0.8, alpha=0.7, label=hist_legend_label,
                         color=label_colors.get(label_key, "purple"))
                has_plotted_data_for_file = True
            elif not_found_or_beyond_max_for_label > 0:
                plt.bar([], [], color=label_colors.get(label_key, "purple"), label=hist_legend_label, alpha=0.7);
                has_plotted_data_for_file = True
        if not has_plotted_data_for_file:
            logger.info(f"No rank data to plot for histogram for file_id '{file_id}'. Skipping plot.");
            plt.close();
            continue
        title_str = f"Histogram of First Relevant Ranks for File: '{file_id}'\n(Total Queries with Ground Truth: {total_queries_in_file_with_gt_and_rank})"
        plt.title(title_str);
        plt.xlabel("Rank of First Relevant Document");
        plt.ylabel("Number of Queries")
        plt.xticks(ticks=list(range(1, MAX_RANK_FOR_HISTOGRAM + 1)));
        plt.xlim(0.5, MAX_RANK_FOR_HISTOGRAM + 0.5)
        plt.minorticks_on();
        plt.grid(axis='y', linestyle='--', alpha=0.7);
        plt.grid(axis='x', linestyle=':', alpha=0.5)
        plt.legend(title="Claim Label (count; >max_rank/not_found)");
        plt.tight_layout()
        plot_filename = output_root / f"ranks_histogram_detailed_{file_id}.png"
        plt.savefig(plot_filename);
        plt.close()
        logger.info(f"  Detailed rank histogram for {file_id} saved to: {plot_filename}")

    logger.info("\n--- Generating Hit@K Bar Charts (per file_id, per label) ---")
    for file_id, hit_data_by_label in hit_at_k_counts.items():
        plt.figure(figsize=(14, 8))
        num_labels = len([lbl for lbl in ORDERED_LABELS_FOR_PLOTTING if lbl in hit_data_by_label])
        if num_labels == 0: logger.info(f"No Hit@K data for {file_id}. Skipping plot."); plt.close(); continue
        bar_width = 0.8 / num_labels if num_labels > 0 else 0.8
        k_indices = np.arange(len(k_values))
        has_plotted_data_for_hit_rate = False
        for i, label_key in enumerate(ORDERED_LABELS_FOR_PLOTTING):
            if label_key not in hit_data_by_label: continue
            rates_for_label = []
            for k_val_plot in k_values:
                counts = hit_data_by_label[label_key].get(k_val_plot, {'hits': 0, 'total': 0})
                rates_for_label.append(counts['hits'] / counts['total'] if counts['total'] > 0 else 0)
            if rates_for_label:
                bar_positions = [x + i * bar_width - (bar_width * (num_labels - 1) / 2) for x in
                                 k_indices[:len(rates_for_label)]]
                total_for_label = hit_data_by_label[label_key].get(k_values[0], {}).get('total', 0)
                plt.bar(bar_positions, rates_for_label, width=bar_width, alpha=0.8,
                        label=f"{label_key} (n={total_for_label})", color=label_colors.get(label_key, "purple"))
                has_plotted_data_for_hit_rate = True
        if not has_plotted_data_for_hit_rate:
            logger.info(f"No data to plot for Hit@K for file_id '{file_id}'. Skipping plot.");
            plt.close();
            continue
        plt.xlabel("K (Top-K Retrieved Documents)");
        plt.ylabel("Hit Rate (Fraction of Queries with >=1 Relevant Document)")
        plt.title(f"Hit Rate @ K for File: '{file_id}'");
        plt.xticks(k_indices, k_values)
        plt.yticks(np.arange(0, 1.1, 0.1));
        plt.ylim(0, 1.05)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title="Claim Label (total queries for label)");
        plt.tight_layout()
        plot_filename = output_root / f"hit_at_k_chart_{file_id}.png"
        plt.savefig(plot_filename);
        plt.close()
        logger.info(f"  Hit@K chart for {file_id} saved to: {plot_filename}")

    logger.info("--- Evaluation process completed. ---")


if __name__ == "__main__":
    if not INPUT_ROOT.exists():
        logger.error(f"CRITICAL: Main input directory for evaluation not found: {INPUT_ROOT}")
        logger.error("Please ensure INPUT_ROOT points to the output of your search script.")
    else:
        logger.info(f"Starting claim-level retrieval evaluation from: {INPUT_ROOT}")
        logger.info(f"Evaluation results will be saved to: {EVAL_OUTPUT_ROOT}")
        logger.info(f"K values for metrics: {K_VALUES}")
        evaluate_retrieval(INPUT_ROOT, EVAL_OUTPUT_ROOT, K_VALUES)
        logger.info("--- Evaluation process completed with detailed analysis. ---")