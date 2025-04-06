# Amazon-BERT: Financial Sentiment Analysis

A high-performance implementation of BERT for financial sentiment analysis. This repository provides tools to train and evaluate BERT-based models for classifying financial text data into sentiment categories (Negative, Neutral, Positive).

## Features

- **High-Performance Training**: Optimized for distributed multi-GPU training with PyTorch DDP
- **Mixed Precision**: Supports automatic mixed precision (AMP) for faster training
- **Distributed Training**: Scales efficiently across multiple GPUs
- **Early Stopping**: Configurable early stopping based on multiple metrics
- **Optimized Evaluation**: Comprehensive model evaluation with detailed metrics and visualizations
- **Gradient Accumulation**: Support for larger effective batch sizes through gradient accumulation
- **Data Caching**: Efficient data loading pipeline with caching for faster training cycles

## Model Architecture

The model architecture consists of a pretrained BERT model with a classification head for sentiment analysis. The implementation includes:

- Pretrained BERT base model
- Dropout for regularization
- Linear classification layer
- Configurable for different numbers of classes

## Requirements

See [requirements.txt](requirements.txt) for the full list of dependencies.

## Usage

### Training

To train the model:

```bash
python train.py
```

By default, this will use all available GPUs. You can configure hyperparameters in the `args` dictionary in the main function.

#### Training Parameters

Key training parameters you can adjust:

- `bert_model_name`: Pretrained BERT model to use (default: "bert-base-uncased")
- `max_length`: Maximum sequence length (default: 512)
- `batch_size`: Batch size per GPU (default: 64)
- `num_epochs`: Maximum number of epochs (default: 50)
- `learning_rate`: Learning rate (default: 2e-5)
- `gradient_accumulation_steps`: Number of steps to accumulate gradients (default: 1)
- `data_fraction`: Fraction of data to use for training (default: 0.1)
- `early_stopping_patience`: Patience for early stopping (default: 5)
- `early_stopping_metric`: Metric to monitor for early stopping (default: 'accuracy')

### Evaluation

To evaluate a trained model:

```bash
./evaluate.sh
```

You can download the pre-trained sentiment analysis models from the following link:
[👉 Download Pre-trained Models on Google Drive](https://drive.google.com/drive/folders/1_wouPl61-PrD_-xzm5OT1nhGp1Q2tlvr?usp=sharing)


After downloading, place the model folder (e.g., saved_models/) in your project root directory or specify the path in the --model_path argument when running the evaluation script or notebook.

Or run the evaluation script directly with custom parameters:

```bash
python evaluate.py \
--model_path ./saved_models \
--bert_model_name bert-base-uncased \
--test_file test.csv \
--batch_size 32 \
--max_length 128 \
--output_dir ./evaluation_results \
--num_workers 4 \
--class_names "Negative,Neutral,Positive"
```

#### Evaluation Metrics

The evaluation script produces comprehensive metrics and visualizations:

- Accuracy, Precision, Recall, F1 Score
- Confusion Matrix
- ROC Curves
- Precision-Recall Curves
- Class Distribution Analysis
- Performance Metrics by Class
- Confidence Histograms
- Misclassification Examples

Results are saved to the specified output directory.

## Dataset

The model is designed to work with financial sentiment datasets in CSV format with the following columns:
- `text`: The financial text content
- `label`: Sentiment label (0 for Negative, 1 for Neutral, 2 for Positive)

## Model Outputs

The trained model can predict sentiment on new financial texts and classifies them as:
- Negative (0)
- Neutral (1)
- Positive (2)

## Performance Optimization

This implementation includes several optimizations:
- Distributed Data Parallel (DDP) training
- Automatic Mixed Precision (AMP)
- Gradient accumulation
- Optimized data loading with caching
- Early stopping to prevent overfitting
- Learning rate scheduling with warmup

## License

[Specify your license here]

## Citation

[If applicable]

## Acknowledgements

- The BERT model is based on the implementation from HuggingFace's Transformers library.
