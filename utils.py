import json
import logging
import os
from io import open
from typing import List, Dict, Optional, Union

import numpy as np
import torch
from pydantic import BaseModel, Field, field_validator 


logger = logging.getLogger(__name__)

def get_device() -> str:
    """Get the current device (GPU or CPU)."""
    device = "cpu"

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"

    return device


class RerankerInputExample(BaseModel):
    """A single example for reranker training/evaluation."""
    guid: str
    query: str  # Natural language query
    code: str   # Code snippet
    label: int  # 1 for relevant, 0 for irrelevant
    query_id: Optional[int] = None  # Used for grouping codes by query during evaluation
    
    class Config:
        arbitrary_types_allowed = True


class PointwiseFeature(BaseModel):
    """Features for pointwise reranking."""
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: List[int]
    label: int
    query_id: Optional[int] = None
    
    class Config:
        arbitrary_types_allowed = True


class PairwiseFeature(BaseModel):
    """Features for pairwise reranking."""
    pos_input_ids: List[int]
    pos_attention_mask: List[int]
    pos_token_type_ids: List[int]
    neg_input_ids: List[int]
    neg_attention_mask: List[int]
    neg_token_type_ids: List[int]
    query_id: Optional[int] = None
    
    class Config:
        arbitrary_types_allowed = True


class ProcessorConfig(BaseModel):
    """Configuration for data processor."""
    data_dir: str = Field(..., description="Data directory path")
    file_name: str = Field(..., description="File name for processing")
    
    @field_validator('data_dir')
    def validate_data_dir(cls, v: str) -> str:
        if not os.path.exists(v):
            raise ValueError(f"Data directory '{v}' does not exist")
        return v


class RerankerProcessor:
    """Processor for code search reranking data."""
    
    def get_train_examples(self, data_dir: str, file_name: str) -> List[RerankerInputExample]:
        """
        Gets training examples.
        
        Args:
            data_dir: Directory containing the data files
            file_name: Name of the training file
            
        Returns:
            List of RerankerInputExample objects
        """
        config = ProcessorConfig(data_dir=data_dir, file_name=file_name)
        return self._create_examples(
            self._read_tsv(os.path.join(config.data_dir, config.file_name)), "train")
    
    def get_dev_examples(self, data_dir: str, file_name: str) -> List[RerankerInputExample]:
        """
        Gets validation examples.
        
        Args:
            data_dir: Directory containing the data files
            file_name: Name of the validation file
            
        Returns:
            List of RerankerInputExample objects
        """
        config = ProcessorConfig(data_dir=data_dir, file_name=file_name)
        return self._create_examples(
            self._read_tsv(os.path.join(config.data_dir, config.file_name)), "dev")
    
    def get_test_examples(self, data_dir: str, file_name: str) -> List[RerankerInputExample]:
        """
        Gets test examples.
        
        Args:
            data_dir: Directory containing the data files
            file_name: Name of the test file
            
        Returns:
            List of RerankerInputExample objects
        """
        config = ProcessorConfig(data_dir=data_dir, file_name=file_name)
        return self._create_examples(
            self._read_tsv(os.path.join(config.data_dir, config.file_name)), "test")
    
    def _read_tsv(self, input_file: str, quotechar: Optional[str] = None) -> List[List[str]]:
        """
        Reads a tab-separated value file.
        
        Args:
            input_file: Path to the input file
            quotechar: Character used for quoting
            
        Returns:
            List of lines, each line is a list of fields
        """
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f.readlines():
                line = line.strip().split('<CODESPLIT>')
                if len(line) < 5:  # Label, query_id, url, query, code at minimum
                    continue
                lines.append(line)
            return lines
        
    def _read_jsonl(self, input_file: str) -> List[Dict]:
        """
        Reads a JSON Lines file.
        
        Args:
            input_file: Path to the input file
            
        Returns:
            List of dictionaries, each representing a JSON object
        """
        examples = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                examples.append(json.loads(line))
        return examples
    
    def _create_examples(self, data: Union[List[List[str]], List[Dict]], set_type: str, is_jsonl: bool = True) -> List[RerankerInputExample]:
        """
        Creates examples for training and evaluation.
        
        Args:
            data: Either a list of lines from TSV file or a list of dictionaries from JSONL file
            set_type: Type of dataset (train, dev, test)
            is_jsonl: Whether the data is in JSONL format
            
        Returns:
            List of RerankerInputExample objects
        """
        examples = []
        
        if is_jsonl:
            # Process JSONL format
            for i, item in enumerate(data):
                guid = f"{set_type}-{i}"
                
                # Extract fields from the JSON object
                label = int(item.get("label", 0))
                query_id = item.get("query_id")
                if query_id.split('_')[-1].isdigit(): # ex. "python_train_123"
                    query_id = int(query_id.split('_')[-1])
                elif query_id.isdigit():
                    query_id = int(query_id)
                else:
                    query_id = i

                query = item.get("query", "")
                code = item.get("code", "")
                
                examples.append(
                    RerankerInputExample(
                        guid=guid,
                        query=query,
                        code=code,
                        label=label,
                        query_id=query_id
                    )
                )
        else:
            # Process TSV format (original implementation)
            for (i, line) in enumerate(data):
                guid = f"{set_type}-{i}"
                
                # Parse line based on the expected format
                if len(line) >= 5:
                    label = int(line[0])
                    query_id = int(line[1]) if line[1].isdigit() else i
                    query = line[3]
                    code = line[4]
                    
                    examples.append(
                        RerankerInputExample(
                            guid=guid,
                            query=query,
                            code=code,
                            label=label,
                            query_id=query_id
                        )
                    )
        
        return examples

def compute_reranker_metrics(scores: np.ndarray, labels: np.ndarray, query_ids: np.ndarray) -> Dict[str, float]:
    """
    Compute evaluation metrics for reranking.
    
    Args:
        scores: Array of model scores
        labels: Array of true labels (1 for relevant, 0 for irrelevant)
        query_ids: Array of query IDs for grouping results
        
    Returns:
        Dictionary of metrics
    """
    assert scores.shape == labels.shape == query_ids.shape  # Ensure correct shapes
    assert len(scores.shape) == 1  # Ensure 1D arrays


    metrics = {}
    
    # Compute classification metrics
    pred_labels = (scores > 0.5).astype(int)
    accuracy = (pred_labels == labels).mean()
    metrics["accuracy"] = float(accuracy)
    
    # If we have query IDs, compute ranking metrics
    if query_ids is not None:
        # Group scores and labels by query
        unique_query_ids = np.unique(query_ids)
        mrr_values = []
        ndcg_values = []
        precision_at_1_values = []
        
        for query_id in unique_query_ids:
            query_mask = query_ids == query_id
            query_scores = scores[query_mask]
            query_labels = labels[query_mask]
            
            if np.sum(query_labels) == 0:
                # Skip queries with no relevant documents
                continue
            
            # Sort by scores in descending order
            sort_indices = np.argsort(-query_scores)
            sorted_labels = query_labels[sort_indices]
            
            # Compute MRR (Mean Reciprocal Rank)
            # Find index of first relevant document (1.0 in labels)
            relevant_indices = np.where(sorted_labels == 1)[0]
            if len(relevant_indices) > 0:
                # MRR = 1 / (rank of first relevant document)
                first_relevant_rank = relevant_indices[0] + 1  # +1 because ranks start at 1
                mrr = 1.0 / first_relevant_rank
                mrr_values.append(mrr)
            
            # Compute NDCG (Normalized Discounted Cumulative Gain)
            # DCG = sum(rel_i / log2(i+1)) for all positions i
            # Ideal DCG = DCG for perfect ranking
            # NDCG = DCG / Ideal DCG
            k = min(10, len(sorted_labels))  # NDCG@10 or fewer if less available
            
            dcg = np.sum(sorted_labels[:k] / np.log2(np.arange(2, k+2)))
            
            # Calculate ideal DCG (sort by relevance)
            ideal_labels = np.sort(query_labels)[::-1]  # Sort in ascending order first and then reverse it
            idcg = np.sum(ideal_labels[:k] / np.log2(np.arange(2, k+2)))
            
            if idcg > 0:
                ndcg = dcg / idcg
                ndcg_values.append(ndcg)
            
            # Compute Precision@1
            precision_at_1 = sorted_labels[0] if len(sorted_labels) > 0 else 0
            precision_at_1_values.append(precision_at_1)
        
        if mrr_values:
            metrics["mrr"] = float(np.mean(mrr_values))
        if ndcg_values:
            metrics["ndcg"] = float(np.mean(ndcg_values))
        if precision_at_1_values:
            metrics["p@1"] = float(np.mean(precision_at_1_values))
    
    return metrics

def format_for_reranking(query, code_snippets, tokenizer, max_length):
    """
    Format query and code snippets for inference with the reranker.
    
    Args:
        query: Natural language query
        code_snippets: List of code snippets to be ranked
        tokenizer: Tokenizer
        max_length: Maximum sequence length
        
    Returns:
        List of features for inference
    """
    features = []
    
    for code in code_snippets:
        # Tokenize query and code
        tokens_a = tokenizer.tokenize(query)
        tokens_b = tokenizer.tokenize(code)
        
        # Account for [CLS], [SEP], [SEP] with "- 3"
        max_tokens = max_length - 3
        
        # Truncate or pad sequences
        if len(tokens_a) + len(tokens_b) > max_tokens:
            # Prioritize code by allocating more tokens to it
            # Allow at least 64 tokens for the query
            query_max = min(64, int(max_tokens * 0.2))
            code_max = max_tokens - min(len(tokens_a), query_max)
            
            if len(tokens_a) > query_max:
                tokens_a = tokens_a[:query_max]
            if len(tokens_b) > code_max:
                tokens_b = tokens_b[:code_max]
        
        # Build token sequence
        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
        tokens = [cls_token] + tokens_a + [sep_token] + tokens_b + [sep_token]
        token_type_ids = [0] + [0] * (len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)
        
        # Convert tokens to IDs
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        
        # Pad sequences
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
        attention_mask = attention_mask + [0] * padding_length
        token_type_ids = token_type_ids + [0] * padding_length
        
        features.append({
            'input_ids': torch.tensor([input_ids], dtype=torch.long),
            'attention_mask': torch.tensor([attention_mask], dtype=torch.long),
            'token_type_ids': torch.tensor([token_type_ids], dtype=torch.long)
        })
    
    return features