# Code Search Reranker

A powerful code search reranker based on CodeBERT that can be trained using either pointwise or pairwise ranking approaches. This project provides a flexible framework for improving code search results by reranking candidate code snippets based on their relevance to natural language queries.

## Features

- Support for both pointwise and pairwise reranking approaches
- Built on top of CodeBERT for robust code understanding
- Efficient data loading and processing pipeline
- Distributed training support
- Comprehensive evaluation metrics (MRR, NDCG, Precision@1)
- TensorBoard integration for training visualization
- Automatic checkpoint saving and best model selection

## Project Structure

```
.
├── dataset_loader.py    # Dataset loading and processing
├── model.py            # CodeBERT reranker model implementation
├── run_reranker.py     # Main training and evaluation script
├── run.sh             # Shell script for dataset preparation and training
└── utils.py           # Utility functions and data structures
```

## Requirements

- PyTorch
- Transformers
- TensorBoardX
- Pydantic

## Installation

1. Clone the repository:
```bash
git clone https://github.com/KnowLangOrg/knowlangbert.git
cd code-search-reranker
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Dataset Preparation
> We assume you have downloaded the dataset from [codesearchnet](https://huggingface.co/datasets/code_search_net    
)

The project expects data in a specific format. Each example should contain:
- Label (1 for relevant, 0 for irrelevant)
- Query ID (for grouping results)
- URL (optional)
- Natural language query
- Code snippet


### Training

You can train the model using the provided shell script:

```bash
# Prepare the dataset
./run.sh prepare

# Train the model
./run.sh train

# Or do both at once
./run.sh all
```

## Model Architecture

The reranker is built on top of CodeBERT and supports two approaches:

1. **Pointwise Reranking**:
   - Treats each query-code pair independently
   - Outputs a relevance score for each pair
   - Uses binary classification loss

2. **Pairwise Reranking**:
   - Considers pairs of code snippets for the same query
   - Uses margin ranking loss to ensure relevant snippets rank higher
   - More effective for ranking tasks

## Evaluation Metrics

The model is evaluated using several metrics:
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)
- Precision@1
- Accuracy

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 