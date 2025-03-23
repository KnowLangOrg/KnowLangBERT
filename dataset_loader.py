import logging
import random
from pathlib import Path
from typing import List, Optional, Dict, Union, Any, Tuple

import torch
from torch.utils.data import ConcatDataset, Dataset
from tqdm import tqdm

from utils import (
    RerankerInputExample,
    RerankerProcessor,
    PointwiseFeature,
    PairwiseFeature,
    get_device
)

logger = logging.getLogger(__name__)


class DatasetLoaderConfig:
    """Configuration for dataset loading."""
    def __init__(
        self,
        data_dir: str,
        language: str,
        dataset_type: str,
        max_seq_length: int,
        tokenizer: Any,
        reranker_type: str = "pointwise",
        shard_id: Optional[int] = None,
        num_shards: Optional[int] = None,
        cache_dir: Optional[str] = None,
        num_workers: int = 4
    ):
        """
        Initialize dataset loader configuration.
        
        Args:
            data_dir: Base directory containing the data files
            language: Programming language (python, java, etc.)
            dataset_type: Dataset type (train, valid, test)
            max_seq_length: Maximum sequence length
            tokenizer: Tokenizer to use
            reranker_type: Reranker type (pointwise or pairwise)
            shard_id: Shard ID for distributed training (optional)
            num_shards: Total number of shards for distributed training (optional)
            cache_dir: Directory to cache processed features (optional)
            num_workers: Number of worker threads for parallel processing
        """
        self.data_dir = data_dir
        self.language = language
        self.dataset_type = dataset_type
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.reranker_type = reranker_type
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.cache_dir = cache_dir
        self.num_workers = num_workers
        
        # Derived paths
        self.dataset_path = Path(data_dir) / language / dataset_type
        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {self.dataset_path}")


class DynamicPairDataset(Dataset):
    """Dataset that dynamically generates pairs for pairwise reranking."""
    
    def __init__(self, examples: List[RerankerInputExample], 
                 tokenizer: Any, 
                 max_seq_length: int,
                 device: str = get_device(),
                 seed: int = 42):
        """
        Initialize dynamic pair dataset.
        
        Args:
            examples: List of examples
            tokenizer: Tokenizer to use
            max_seq_length: Maximum sequence length
            seed: Random seed for reproducibility
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.seed = seed
        self.device = device
        
        # Group examples by query_id
        self.query_to_examples = self._group_by_query()
        
        # Setup indices for efficient retrieval
        self.pos_indices = []
        for query_id, data in self.query_to_examples.items():
            if len(data['pos']) > 0 and len(data['neg']) > 0:
                for i, _ in enumerate(data['pos']):
                    self.pos_indices.append((query_id, i))
        
        # Set random seed for reproducibility
        random.seed(self.seed)
        
    def _group_by_query(self) -> Dict[int, Dict[str, List[RerankerInputExample]]]:
        """Group examples by query ID."""
        query_dict = {}
        for ex in self.examples:
            if ex.query_id not in query_dict:
                query_dict[ex.query_id] = {'pos': [], 'neg': []}
            
            if ex.label == 1:
                query_dict[ex.query_id]['pos'].append(ex)
            else:
                query_dict[ex.query_id]['neg'].append(ex)
        
        # Filter out queries with no positives or no negatives
        return {qid: data for qid, data in query_dict.items() 
                if len(data['pos']) > 0 and len(data['neg']) > 0}
    
    def __len__(self) -> int:
        """Get dataset length (number of positive examples with available negatives)."""
        return len(self.pos_indices)
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Get a pair of examples at index.
        
        Args:
            idx: Index
            
        Returns:
            Tuple of tensors for pairwise training
        """
        query_id, pos_idx = self.pos_indices[idx]
        query_data = self.query_to_examples[query_id]
        
        # Get positive example
        pos_ex : RerankerInputExample = query_data['pos'][pos_idx]
        
        # Sample a negative example
        neg_ex : RerankerInputExample = random.choice(query_data['neg'])
        
        # Convert query and positive example to features
        pos_tokens_query = self.tokenizer.tokenize(pos_ex.query)
        pos_tokens_code = self.tokenizer.tokenize(pos_ex.code)
        
        # Convert query and negative example to features
        neg_tokens_query = self.tokenizer.tokenize(neg_ex.query)
        neg_tokens_code = self.tokenizer.tokenize(neg_ex.code)
        
        # Account for [CLS], [SEP], [SEP] with "- 3" for both examples
        max_tokens = self.max_seq_length - 3
        
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
        cls_token = self.tokenizer.cls_token
        sep_token = self.tokenizer.sep_token
        pos_tokens = [cls_token] + pos_tokens_query + [sep_token] + pos_tokens_code + [sep_token]
        pos_token_type_ids = [0] + [0] * (len(pos_tokens_query) + 1) + [1] * (len(pos_tokens_code) + 1)
        
        neg_tokens = [cls_token] + neg_tokens_query + [sep_token] + neg_tokens_code + [sep_token]
        neg_token_type_ids = [0] + [0] * (len(neg_tokens_query) + 1) + [1] * (len(neg_tokens_code) + 1)
        
        # Convert tokens to IDs
        pos_input_ids = self.tokenizer.convert_tokens_to_ids(pos_tokens)
        pos_attention_mask = [1] * len(pos_input_ids)
        
        neg_input_ids = self.tokenizer.convert_tokens_to_ids(neg_tokens)
        neg_attention_mask = [1] * len(neg_input_ids)
        
        # Pad sequences
        pos_padding_length = self.max_seq_length - len(pos_input_ids)
        pos_input_ids = pos_input_ids + [self.tokenizer.pad_token_id] * pos_padding_length
        pos_attention_mask = pos_attention_mask + [0] * pos_padding_length
        pos_token_type_ids = pos_token_type_ids + [0] * pos_padding_length
        
        neg_padding_length = self.max_seq_length - len(neg_input_ids)
        neg_input_ids = neg_input_ids + [self.tokenizer.pad_token_id] * neg_padding_length
        neg_attention_mask = neg_attention_mask + [0] * neg_padding_length
        neg_token_type_ids = neg_token_type_ids + [0] * neg_padding_length
        
        return (
            torch.tensor(pos_input_ids, dtype=torch.long, device=self.device),
            torch.tensor(pos_attention_mask, dtype=torch.long, device=self.device),
            torch.tensor(pos_token_type_ids, dtype=torch.long, device=self.device),
            torch.tensor(neg_input_ids, dtype=torch.long, device=self.device),
            torch.tensor(neg_attention_mask, dtype=torch.long, device=self.device),
            torch.tensor(neg_token_type_ids, dtype=torch.long, device=self.device),
            torch.tensor(1, dtype=torch.long, device=self.device)  # Dummy label
        )


class PointwiseDataset(Dataset):
    """Dataset for pointwise reranking."""
    
    def __init__(self, examples: List[RerankerInputExample], 
                 tokenizer: Any, 
                 max_seq_length: int,
                 device: str = get_device()):
        """
        Initialize pointwise dataset.
        
        Args:
            examples: List of examples
            tokenizer: Tokenizer to use
            max_seq_length: Maximum sequence length
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.device = device

        
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Get an example at index.
        
        Args:
            idx: Index
            
        Returns:
            Tuple of tensors for pointwise training
        """
        example = self.examples[idx]
        
        # Tokenize query and code
        tokens_a = self.tokenizer.tokenize(example.query)
        tokens_b = self.tokenizer.tokenize(example.code)
        
        # Account for [CLS], [SEP], [SEP] with "- 3"
        max_tokens = self.max_seq_length - 3
        
        # Truncate or pad sequences
        if len(tokens_a) + len(tokens_b) > max_tokens:
            # Prioritize code by allocating more tokens to it
            query_max = min(64, int(max_tokens * 0.2))
            code_max = max_tokens - min(len(tokens_a), query_max)
            
            if len(tokens_a) > query_max:
                tokens_a = tokens_a[:query_max]
            if len(tokens_b) > code_max:
                tokens_b = tokens_b[:code_max]
        
        # Build token sequence
        cls_token = self.tokenizer.cls_token
        sep_token = self.tokenizer.sep_token
        tokens = [cls_token] + tokens_a + [sep_token] + tokens_b + [sep_token]
        token_type_ids = [0] + [0] * (len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)
        
        # Convert tokens to IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        
        # Pad sequences
        padding_length = self.max_seq_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
        attention_mask = attention_mask + [0] * padding_length
        token_type_ids = token_type_ids + [0] * padding_length
        
        return (
            torch.tensor(input_ids, dtype=torch.long, device=self.device),
            torch.tensor(attention_mask, dtype=torch.long, device=self.device),
            torch.tensor(token_type_ids, dtype=torch.long, device=self.device),
            torch.tensor(example.label, dtype=torch.long, device=self.device),
            torch.tensor(example.query_id, dtype=torch.long, device=self.device) if example.query_id is not None else torch.tensor(0, dtype=torch.long, device=self.device)
        )


def get_dataset_files(config: DatasetLoaderConfig) -> List[Path]:
    """
    Get all dataset files for the given configuration.
    
    Args:
        config: Dataset loader configuration
        
    Returns:
        List of dataset file paths
    """
    pattern = f"{config.language}_{config.dataset_type}_*.txt"
    files = list(config.dataset_path.glob(pattern))
    
    # Sort files numerically by the index number
    def get_file_index(file_path):
        try:
            # Extract the numeric part (e.g., "0" from "python_train_0.txt")
            index_str = file_path.stem.split("_")[-1]
            return int(index_str)
        except (ValueError, IndexError):
            # If parsing fails, return a large number to sort these files last
            return float('inf')
    
    files = sorted(files, key=get_file_index)
    
    if not files:
        raise ValueError(f"No data files found at {config.dataset_path} matching {pattern}")
    
    # If using sharding for distributed training, select only appropriate files
    if config.shard_id is not None and config.num_shards is not None:
        files = [files[i] for i in range(len(files)) if i % config.num_shards == config.shard_id]
    
    return files


def load_examples(config: DatasetLoaderConfig, max_files: int = 6) -> List[RerankerInputExample]:
    """
    Load examples from dataset files.
    
    Args:
        config: Dataset loader configuration
        max_files: Maximum number of files to load (for memory efficiency)
        
    Returns:
        List of examples
    """
    # Get dataset files
    files = get_dataset_files(config)
    logger.info(f"Found {len(files)} data files for {config.language}/{config.dataset_type}")
    
    # Limit the number of files for memory efficiency
    files = files[:max_files]
    
    # Create processor
    processor = RerankerProcessor()
    
    # Load examples from files
    examples = []
    for file_path in tqdm(files, desc=f"Loading {config.dataset_type} examples"):
        examples.extend(processor._create_examples(
            processor._read_tsv(str(file_path)), 
            set_type=config.dataset_type
        ))
    
    logger.info(f"Loaded {len(examples)} examples from {len(files)} files")
    return examples


def load_datasets(config: DatasetLoaderConfig, device : str = get_device()) -> Union[Dataset, ConcatDataset]:
    """
    Load and process all data files for the given configuration.
    
    Args:
        config: Dataset loader configuration
        
    Returns:
        Dataset for training or evaluation
    """
    # Load examples from files
    examples = load_examples(config)

    # Create dataset based on reranker type
    if config.reranker_type == "pointwise":
        return PointwiseDataset(examples, config.tokenizer, config.max_seq_length, device)
    elif config.reranker_type == "pairwise":
        return DynamicPairDataset(
            examples, 
            config.tokenizer, 
            config.max_seq_length,
            device=device
        )
    else:
        raise ValueError(f"Invalid reranker type: {config.reranker_type}")