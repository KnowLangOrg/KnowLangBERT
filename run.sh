#!/bin/bash
# Script to prepare datasets and train CodeBERT reranker

# Configuration
DATASET_ROOT="./dataset"                       # Adjust to your CodeSearchNet dataset location
OUTPUT_ROOT="./output"                         # Adjust to your desired output location
LANGUAGE="python"                              # Language to process
MODEL_NAME="microsoft/codebert-base"           # Base model to finetune
NUM_NEGATIVES=5                                # Number of negative examples per positive
NUM_WORKERS=8                                  # Number of worker threads for data preparation
BATCH_SIZE=32                                  # Training batch size
LEARNING_RATE=2e-5                             # Learning rate
NUM_EPOCHS=1                                  # Number of training epochs
RERANKER_TYPE="pairwise"                       # Reranker type (pointwise or pairwise)

# Create directories
RERANKER_DATA_DIR="${OUTPUT_ROOT}/reranker_dataset"
MODEL_OUTPUT_DIR="${OUTPUT_ROOT}/reranker_models/${LANGUAGE}"
CACHE_DIR="${OUTPUT_ROOT}/cache"

mkdir -p "${RERANKER_DATA_DIR}"
mkdir -p "${MODEL_OUTPUT_DIR}"
mkdir -p "${CACHE_DIR}"

# Function to prepare datasets
prepare_dataset() {
    echo "============================================"
    echo "Preparing reranker datasets for ${LANGUAGE}"
    echo "============================================"

    python prepare_reranker_dataset.py \
      --input_dir "${DATASET_ROOT}" \
      --output_dir "${RERANKER_DATA_DIR}" \
      --language "${LANGUAGE}" \
      --num_negatives "${NUM_NEGATIVES}" \
      --num_workers "${NUM_WORKERS}" \
      --dataset_types train valid test
}

# Function to train model
train_model() {
    echo "============================================"
    echo "Training reranker model for ${LANGUAGE}"
    echo "============================================"

    python run_reranker.py \
      --data_dir "${RERANKER_DATA_DIR}" \
      --language "${LANGUAGE}" \
      --model_type "roberta" \
      --model_name_or_path "${MODEL_NAME}" \
      --output_dir "${MODEL_OUTPUT_DIR}" \
      --reranker_type "${RERANKER_TYPE}" \
      --max_seq_length 256 \
      --cache_dir "${CACHE_DIR}" \
      --per_gpu_train_batch_size "${BATCH_SIZE}" \
      --per_gpu_eval_batch_size "${BATCH_SIZE}" \
      --learning_rate "${LEARNING_RATE}" \
      --num_train_epochs "${NUM_EPOCHS}" \
      --warmup_steps 1000 \
      --logging_steps 100 \
      --save_steps 1000 \
      --seed 42 \
      --do_train \
      --do_eval \
      --evaluate_during_training

    echo "============================================"
    echo "Training completed!"
    echo "Model saved to: ${MODEL_OUTPUT_DIR}"
    echo "============================================"
}

# Check command line arguments and execute the requested step
if [ $# -eq 0 ]; then
    echo "Usage: $0 [prepare|train|all]"
    echo "  prepare: Prepare the dataset only"
    echo "  train: Train the model only"
    echo "  all: Prepare dataset and train model"
    exit 1
fi

case "$1" in
    "prepare")
        prepare_dataset
        ;;
    "train")
        train_model
        ;;
    "all")
        prepare_dataset
        train_model
        ;;
    *)
        echo "Invalid option: $1"
        echo "Usage: $0 [prepare|train|all]"
        exit 1
        ;;
esac