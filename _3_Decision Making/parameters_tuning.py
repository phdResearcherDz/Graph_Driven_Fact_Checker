import json
from pathlib import Path
import optuna
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

# --- Define Models, Data, and Evidence Types to Loop Through ---
MODELS_TO_TUNE = [
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    # "emilyalsentzer/Bio_ClinicalBERT",
    # "dmis-lab/biobert-v1.1",
    # "michiyasunaga/BioLinkBERT-base"
]

DATA_ROOT = Path("../prepared_training_data_json_full_content")
RESULTS_ROOT = Path("./tuning_results_full_content")  # Main results folder

EVIDENCE_TYPES_TO_TUNE = ["primary_only"] #, "secondary_only", "both_sources"

try:
    # Automatically discover datasets from the directory structure
    DATASETS_TO_TUNE = [d.name for d in (DATA_ROOT / EVIDENCE_TYPES_TO_TUNE[0]).iterdir() if d.is_dir()]
except FileNotFoundError:
    logger.error(f"Could not find evidence folders in {DATA_ROOT}. Please run the data preparation script first.")
    DATASETS_TO_TUNE = []

# --- Tuning Configuration ---
TUNING_CONFIG = {
    "n_trials": 20,  # Number of different hyperparameter combinations to try
    "direction": "maximize",
}


# --- 2. HELPER FUNCTIONS (UNCHANGED) ---

def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


# --- 3. REFACTORED TUNING LOGIC ---

def tune_single_configuration(model_name: str, dataset_name: str, evidence_type: str):
    """
    Performs a full hyperparameter search for a single combination of
    model, dataset, and evidence type.
    """
    # Sanitize model name for directory path
    sanitized_model_name = model_name.replace('/', '_')

    logger.info("=" * 80)
    logger.info(f"--- Starting New Tuning Run ---")
    logger.info(f"Model: '{model_name}'")
    logger.info(f"Dataset: '{dataset_name}'")
    logger.info(f"Evidence: '{evidence_type}'")
    logger.info("=" * 80)

    # a. Define dynamic output directory for this specific run
    output_dir = RESULTS_ROOT / sanitized_model_name / dataset_name / evidence_type
    if output_dir.exists() and any(output_dir.iterdir()):
        logger.warning(f"Output directory {output_dir} is not empty. Deleting content to ensure a clean run.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to: {output_dir}")

    # b. Load and prepare data
    train_file = DATA_ROOT / evidence_type / dataset_name / "train.json"
    dev_file = DATA_ROOT / evidence_type / dataset_name / "dev.json"

    if not train_file.exists() or not dev_file.exists():
        logger.error(f"Data files not found for {dataset_name} ({evidence_type}). Skipping run.")
        return

    dataset = load_dataset('json', data_files={'train': str(train_file), 'validation': str(dev_file)})
    labels = dataset["train"].unique("label")
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess_function(examples):
        tokenized_inputs = tokenizer(
            examples["claim_text"],
            examples["evidence"],
            truncation="only_second",
            padding="max_length",
            max_length=512,
        )
        tokenized_inputs["label"] = [label2id[label] for label in examples["label"]]
        return tokenized_inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
    logger.info("Data loaded and tokenized successfully.")

    # c. Define model initializer using the 'model_name' parameter
    def model_init(trial):
        return AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )

    # d. Set up Trainer
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        disable_tqdm=False,
        save_total_limit=1,  # Only keep the single best checkpoint
        report_to="none"  # Disable integrations like wandb
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    def optuna_hp_space(trial: optuna.Trial) -> dict:
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 5),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16]),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
            "warmup_ratio": trial.suggest_categorical("warmup_ratio", [0.0, 0.1]),
        }

    # e. Run the search
    logger.info("Starting hyperparameter search...")
    best_run = trainer.hyperparameter_search(
        hp_space=optuna_hp_space,
        n_trials=TUNING_CONFIG["n_trials"],
        direction=TUNING_CONFIG["direction"],
    )

    logger.info(f"--- Hyperparameter Search Complete for {model_name} on {dataset_name} ({evidence_type}) ---")
    logger.info(f"Best trial F1-score: {best_run.objective}")
    logger.info(f"Best hyperparameters: {best_run.hyperparameters}")

    # f. Save the best parameters
    best_params_path = output_dir / "best_hyperparameters.json"
    with open(best_params_path, "w") as f:
        json.dump(best_run.hyperparameters, f, indent=4)
    logger.info(f"✅ Best hyperparameters saved to: {best_params_path}\n")


if __name__ == "__main__":
    logger.info("Starting FULL automated hyperparameter tuning pipeline.")

    if not DATASETS_TO_TUNE:
        logger.warning("No datasets found to process. Exiting.")
    else:
        # The main triple-nested loop to run all experiments
        for model_name in MODELS_TO_TUNE:
            for evidence_type in EVIDENCE_TYPES_TO_TUNE:
                for dataset_name in DATASETS_TO_TUNE:
                    try:
                        tune_single_configuration(
                            model_name=model_name,
                            dataset_name=dataset_name,
                            evidence_type=evidence_type
                        )
                    except Exception as e:
                        logger.error(f"FATAL ERROR during run for {model_name}/{dataset_name}/{evidence_type}.")
                        logger.error(e, exc_info=True)
                        continue  # Continue to the next job

    logger.info("🎉🎉🎉 FULL automated tuning pipeline finished. 🎉🎉🎉")