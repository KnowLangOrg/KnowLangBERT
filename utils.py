import csv
import logging
import os
from io import open
from typing import List, Dict, Optional, Tuple, Any, Union, Sequence

import numpy as np
import torch
from pydantic import BaseModel, Field, field_validator 


logger = logging.getLogger(__name__)


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
    
    def _create_examples(self, lines: List[List[str]], set_type: str) -> List[RerankerInputExample]:
        """
        Creates examples for training and evaluation.
        
        Args:
            lines: List of lines from the input file
            set_type: Type of dataset (train, dev, test)
            
        Returns:
            List of RerankerInputExample objects
        """
        examples = []
        for (i, line) in enumerate(lines):
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

def convert_examples_to_features(examples: List[RerankerInputExample], 
                                max_length: int,
                                tokenizer,
                                reranker_type: str = "pointwise",
                                cls_token: str = "[CLS]",
                                sep_token: str = "[SEP]",
                                pad_token: int = 0,
                                cls_token_segment_id: int = 0,
                                pad_token_segment_id: int = 0) -> List[Union[PointwiseFeature, PairwiseFeature]]:
    """
    Convert examples to features compatible with the model.
    
    For pointwise reranking: each example is a (query, code) pair with a relevance label
    For pairwise reranking: examples are grouped by query_id, and each group generates 
                            pairs of (relevant, irrelevant) code snippets
    """
    features = []
    
    if reranker_type == "pointwise":
        # Process each example independently
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info(f"Writing pointwise example {ex_index} of {len(examples)}")
            
            # Tokenize query and code
            tokens_a = tokenizer.tokenize(example.query)
            tokens_b = tokenizer.tokenize(example.code)
            
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
            tokens = [cls_token] + tokens_a + [sep_token] + tokens_b + [sep_token]
            token_type_ids = [cls_token_segment_id] + [0] * (len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)
            
            # Convert tokens to IDs
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            attention_mask = [1] * len(input_ids)
            
            # Pad sequences
            padding_length = max_length - len(input_ids)
            input_ids = input_ids + [pad_token] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            token_type_ids = token_type_ids + [pad_token_segment_id] * padding_length
            
            assert len(input_ids) == max_length
            assert len(attention_mask) == max_length
            assert len(token_type_ids) == max_length
            
            # Create feature
            features.append(
                PointwiseFeature(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    label=example.label,
                    query_id=example.query_id
                )
            )
    
    else:  # pairwise reranking
        # Group examples by query_id
        query_to_examples = {}
        for example in examples:
            if example.query_id not in query_to_examples:
                query_to_examples[example.query_id] = []
            query_to_examples[example.query_id].append(example)
        
        pair_count = 0
        # For each query, create positive-negative pairs
        for query_id, query_examples in query_to_examples.items():
            positive_examples = [ex for ex in query_examples if ex.label == 1]
            negative_examples = [ex for ex in query_examples if ex.label == 0]
            
            # Skip if no positive or negative examples
            if not positive_examples or not negative_examples:
                continue
            
            # Create pairs (each positive example paired with each negative example)
            for pos_example in positive_examples:
                for neg_example in negative_examples:
                    pair_count += 1
                    if pair_count % 10000 == 0:
                        logger.info(f"Writing pairwise example {pair_count}")
                    
                    # Process positive example
                    pos_tokens_query = tokenizer.tokenize(pos_example.query)
                    pos_tokens_code = tokenizer.tokenize(pos_example.code)
                    
                    # Process negative example (same query, different code)
                    neg_tokens_query = tokenizer.tokenize(neg_example.query)
                    neg_tokens_code = tokenizer.tokenize(neg_example.code)
                    
                    # Account for [CLS], [SEP], [SEP] with "- 3" for both examples
                    max_tokens = max_length - 3
                    
                    # Truncate positive example
                    if len(pos_tokens_query) + len(pos_tokens_code) > max_tokens:
                        query_max = min(64, int(max_tokens * 0.2))
                        code_max = max_tokens - min(len(pos_tokens_query), query_max)
                        
                        if len(pos_tokens_query) > query_max:
                            pos_tokens_query = pos_tokens_query[:query_max]
                        if len(pos_tokens_code) > code_max:
                            pos_tokens_code = pos_tokens_code[:code_max]
                    
                    # Truncate negative example
                    if len(neg_tokens_query) + len(neg_tokens_code) > max_tokens:
                        query_max = min(64, int(max_tokens * 0.2))
                        code_max = max_tokens - min(len(neg_tokens_query), query_max)
                        
                        if len(neg_tokens_query) > query_max:
                            neg_tokens_query = neg_tokens_query[:query_max]
                        if len(neg_tokens_code) > code_max:
                            neg_tokens_code = neg_tokens_code[:code_max]
                    
                    # Build token sequences
                    pos_tokens = [cls_token] + pos_tokens_query + [sep_token] + pos_tokens_code + [sep_token]
                    pos_token_type_ids = [cls_token_segment_id] + [0] * (len(pos_tokens_query) + 1) + [1] * (len(pos_tokens_code) + 1)
                    
                    neg_tokens = [cls_token] + neg_tokens_query + [sep_token] + neg_tokens_code + [sep_token]
                    neg_token_type_ids = [cls_token_segment_id] + [0] * (len(neg_tokens_query) + 1) + [1] * (len(neg_tokens_code) + 1)
                    
                    # Convert tokens to IDs
                    pos_input_ids = tokenizer.convert_tokens_to_ids(pos_tokens)
                    pos_attention_mask = [1] * len(pos_input_ids)
                    
                    neg_input_ids = tokenizer.convert_tokens_to_ids(neg_tokens)
                    neg_attention_mask = [1] * len(neg_input_ids)
                    
                    # Pad sequences
                    pos_padding_length = max_length - len(pos_input_ids)
                    pos_input_ids = pos_input_ids + [pad_token] * pos_padding_length
                    pos_attention_mask = pos_attention_mask + [0] * pos_padding_length
                    pos_token_type_ids = pos_token_type_ids + [pad_token_segment_id] * pos_padding_length
                    
                    neg_padding_length = max_length - len(neg_input_ids)
                    neg_input_ids = neg_input_ids + [pad_token] * neg_padding_length
                    neg_attention_mask = neg_attention_mask + [0] * neg_padding_length
                    neg_token_type_ids = neg_token_type_ids + [pad_token_segment_id] * neg_padding_length
                    
                    assert len(pos_input_ids) == max_length
                    assert len(pos_attention_mask) == max_length
                    assert len(pos_token_type_ids) == max_length
                    
                    assert len(neg_input_ids) == max_length
                    assert len(neg_attention_mask) == max_length
                    assert len(neg_token_type_ids) == max_length
                    
                    # Create pairwise feature
                    features.append(
                        PairwiseFeature(
                            pos_input_ids=pos_input_ids,
                            pos_attention_mask=pos_attention_mask,
                            pos_token_type_ids=pos_token_type_ids,
                            neg_input_ids=neg_input_ids,
                            neg_attention_mask=neg_attention_mask,
                            neg_token_type_ids=neg_token_type_ids,
                            query_id=query_id
                        )
                    )
    
    return features

def compute_reranker_metrics(scores: np.ndarray, labels: np.ndarray, query_ids: np.ndarray = None) -> Dict[str, float]:
    """
    Compute evaluation metrics for reranking.
    
    Args:
        scores: Array of model scores
        labels: Array of true labels (1 for relevant, 0 for irrelevant)
        query_ids: Array of query IDs for grouping results
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Compute classification metrics
    pred_labels = (scores > 0.5).astype(int) if scores.ndim == 1 else np.argmax(scores, axis=1)
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
            ideal_labels = np.sort(query_labels)[::-1]  # Sort in descending order
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