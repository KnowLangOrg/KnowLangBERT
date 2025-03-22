# Create output directory
mkdir -p ./reranker_models/python

# Set parameters
DATA_DIR="../data/codesearch"  # Adjust to your data directory
LANG=python
MODEL_TYPE=roberta
MODEL_NAME_OR_PATH=microsoft/codebert-base
OUTPUT_DIR=./reranker_models/$LANG
RERANKER_TYPE=pointwise  # Options: pointwise, pairwise

# Run training
python run_reranker.py \
  --model_type $MODEL_TYPE \
  --reranker_type $RERANKER_TYPE \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --data_dir $DATA_DIR/train_valid/$LANG \
  --train_file train.txt \
  --dev_file valid.txt \
  --output_dir $OUTPUT_DIR \
  --max_seq_length 256 \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 5 \
  --warmup_steps 1000 \
  --logging_steps 500 \
  --save_steps 1000 \
  --seed 42 \
  --do_train \
  --do_eval \
  --evaluate_during_training

# Evaluate on test set
python run_reranker.py \
  --model_type $MODEL_TYPE \
  --reranker_type $RERANKER_TYPE \
  --model_name_or_path $OUTPUT_DIR/checkpoint-best \
  --data_dir $DATA_DIR/test/$LANG \
  --dev_file batch_0.txt \
  --output_dir $OUTPUT_DIR/test_results \
  --max_seq_length 256 \
  --per_gpu_eval_batch_size 32 \
  --do_eval