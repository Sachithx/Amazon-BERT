#!/bin/bash

# Create directories
mkdir -p ./evaluation_results

# Run the evaluation script with appropriate parameters
python evaluate.py \
    --model_path ./saved_models \
    --bert_model_name bert-base-uncased \
    --test_file test.csv \
    --batch_size 512 \
    --max_length 512 \
    --output_dir ./evaluation_results \
    --num_workers 2 \
    --class_names "Negative,Neutral,Positive" \
    --fraction 0.5

echo "Evaluation complete. Results saved to ./evaluation_results/"