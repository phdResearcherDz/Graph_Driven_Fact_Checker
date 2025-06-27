import json
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
import shutil

# --- 1. CONFIGURATION ---

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Define Models, Data, and Paths to Loop Through ---
MODELS_TO_EVALUATE = [
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    # "emilyalsentzer/Bio_ClinicalBERT",
    # "dmis-lab/biobert-v1.1",
    # "michiyasunaga/BioLinkBERT-base"
]

DATA_ROOT = Path("../prepared_training_data_json_full_content")
TUNING_RESULTS_ROOT = Path("./tuning_results")
FINAL_RESULTS_ROOT = Path("./final_models_and_evaluations_full_content")

EVIDENCE_TYPES_TO_EVALUATE = ["primary_only"]
try:
    DATASETS_TO_EVALUATE = [d.name for d in (DATA_ROOT / EVIDENCE_TYPES_TO_EVALUATE[0]).iterdir() if d.is_dir()]
    # DATASETS_TO_EVALUATE = ["SciFact"]
except FileNotFoundError:
    logger.error(f"Could not find evidence folders in {DATA_ROOT}. Please run the data preparation script first.")
    DATASETS_TO_EVALUATE = []


# --- 2. HELPER FUNCTIONS (UNCHANGED) ---

def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


# --- 3. FINAL TRAINING & EVALUATION LOGIC (MODIFIED FOR BEFORE/AFTER EVAL) ---

def train_and_evaluate_single_configuration(model_name: str, dataset_name: str, evidence_type: str):
    sanitized_model_name = model_name.replace('/', '_')
    is_scifact_blind_test = (dataset_name == "SciFact")

    logger.info("=" * 80)
    logger.info(f"--- Starting Final Run ---")
    logger.info(f"Model: '{model_name}', Dataset: '{dataset_name}', Evidence: '{evidence_type}'")
    logger.info("=" * 80)

    # a. Define paths
    params_path = TUNING_RESULTS_ROOT / sanitized_model_name / dataset_name / evidence_type / "best_hyperparameters.json"
    output_dir = FINAL_RESULTS_ROOT / sanitized_model_name / dataset_name / evidence_type
    output_dir.mkdir(parents=True, exist_ok=True)

    # b. Load best hyperparameters
    if not params_path.exists():
        logger.error(f"Best hyperparameters file not found at {params_path}. Skipping run.")
        return
    with open(params_path, 'r') as f:
        best_params = json.load(f)
    logger.info(f"Loaded best hyperparameters: {best_params}")

    if dataset_name == "Scifact":
        test = "dev.json"
    else:
        test = "test.json"
    # c. Load and prepare all data splits
    data_files = {'train': str(DATA_ROOT / evidence_type / dataset_name / "train.json"),
                  'validation': str(DATA_ROOT / evidence_type / dataset_name / "dev.json"),
                  'test': str(DATA_ROOT / evidence_type / dataset_name / test)}
    dataset = load_dataset('json', data_files=data_files)
    labels = dataset["train"].unique("label")
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess_function(examples):
        tokenized_inputs = tokenizer(
            examples["claim_text"], examples["evidence"],
            truncation="only_second", padding="max_length", max_length=512
        )
        if "label" in examples:
            tokenized_inputs["label"] = [label2id[label] for label in examples["label"]]
        return tokenized_inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    if is_scifact_blind_test and "label" in tokenized_datasets["test"].features:
        tokenized_datasets["test"] = tokenized_datasets["test"].remove_columns(["label"])
    logger.info("All data loaded and tokenized.")

    # d. Initialize the Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels), id2label=id2label, label2id=label2id
    )
    logger.info("Base model loaded.")

    # --- NEW: Step 1 - Evaluate the base model BEFORE fine-tuning ---
    logger.info("--- Evaluating base model (zero-shot performance)... ---")
    # We need a temporary trainer with minimal args just for evaluation
    base_model_trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=output_dir / "base_model_eval", report_to="none"),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    eval_dataset = tokenized_datasets["validation"] if is_scifact_blind_test else tokenized_datasets["test"]
    eval_set_name = "DEV" if is_scifact_blind_test else "TEST"

    base_model_metrics = base_model_trainer.evaluate(eval_dataset=eval_dataset)
    logger.info(f"--- Base Model Performance on {eval_set_name} Set ---")
    for key, value in base_model_metrics.items():
        logger.info(f"  {key.replace('eval_', '')}: {value:.4f}")

    # Save the "before" metrics
    base_metrics_path = output_dir / f"metrics_before_tuning_{eval_set_name.lower()}.json"
    with open(base_metrics_path, "w") as f:
        json.dump(base_model_metrics, f, indent=4)
    logger.info(f"✅ Base model metrics saved to: {base_metrics_path}")
    # --- END OF NEW SECTION ---

    # --- Step 2: Proceed with Fine-Tuning ---
    # e. Set up Training Arguments using the best hyperparameters
    training_args = TrainingArguments(
        output_dir=str(output_dir), **best_params,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1",
        save_total_limit=1, report_to="none"
    )

    # f. Initialize the main Trainer
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer, compute_metrics=compute_metrics,
    )

    # g. Train the model
    logger.info("\n--- Starting fine-tuning run... ---")
    trainer.train()
    logger.info("Fine-tuning complete.")

    # h. Evaluate the fine-tuned model
    logger.info(f"--- Evaluating fine-tuned model on {eval_set_name} set... ---")
    if is_scifact_blind_test:
        final_metrics = trainer.evaluate(eval_dataset=tokenized_datasets["validation"])
        logger.info(f"--- Fine-Tuned Model Performance on DEV Set (SciFact) ---")
        for key, value in final_metrics.items():
            logger.info(f"  {key.replace('eval_', '')}: {value:.4f}")

        metrics_path = output_dir / f"metrics_after_tuning_dev.json"
        with open(metrics_path, "w") as f:
            json.dump(final_metrics, f, indent=4)
        logger.info(f"✅ Fine-tuned dev set metrics saved to: {metrics_path}")

        logger.info("Generating predictions for the blind TEST set...")
        test_predictions = trainer.predict(test_dataset=tokenized_datasets["test"])
        predicted_labels = [id2label[pid] for pid in np.argmax(test_predictions.predictions, axis=1)]
        prediction_results = [{"claim_id": r["claim_id"], "predicted_label": lbl} for r, lbl in
                              zip(dataset["test"], predicted_labels)]
        predictions_path = output_dir / "test_set_predictions.json"
        with open(predictions_path, "w") as f:
            json.dump(prediction_results, f, indent=4)
        logger.info(f"✅ Test set predictions saved to: {predictions_path}")
    else:
        final_metrics = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
        logger.info("--- Fine-Tuned Model Performance on TEST Set ---")
        for key, value in final_metrics.items():
            logger.info(f"  {key.replace('eval_', '')}: {value:.4f}")
        metrics_path = output_dir / "metrics_after_tuning_test.json"
        with open(metrics_path, "w") as f:
            json.dump(final_metrics, f, indent=4)
        logger.info(f"✅ Fine-tuned test set metrics saved to: {metrics_path}")

    # i. Save the final model
    final_model_path = output_dir / "final_model"
    trainer.save_model(final_model_path)
    logger.info(f"✅ Final fine-tuned model saved to: {final_model_path}\n")


if __name__ == "__main__":
    logger.info("Starting FINAL automated 'before and after' training and evaluation pipeline.")
    if not DATASETS_TO_EVALUATE:
        logger.warning("No datasets found to process. Exiting.")
    else:
        for model_name in MODELS_TO_EVALUATE:
            for evidence_type in EVIDENCE_TYPES_TO_EVALUATE:
                for dataset_name in DATASETS_TO_EVALUATE:
                    try:
                        train_and_evaluate_single_configuration(
                            model_name=model_name, dataset_name=dataset_name, evidence_type=evidence_type
                        )
                    except Exception as e:
                        logger.error(f"FATAL ERROR during run for {model_name}/{dataset_name}/{evidence_type}.")
                        logger.error(e, exc_info=True)
                        continue
    logger.info("🎉🎉🎉 FULL automated pipeline finished. 🎉🎉🎉")