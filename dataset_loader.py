import os
import glob
import logging
from pathlib import Path
from typing import List, Optional, Dict, Union, Any, Tuple
import concurrent.futures

import torch
from torch.utils.data import TensorDataset, ConcatDataset
from tqdm import tqdm

from utils import (
    RerankerInputExample,
    RerankerProcessor,
    convert_examples_to_features,
    PointwiseFeature,
    PairwiseFeature
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


def get_dataset_files(config: DatasetLoaderConfig) -> List[Path]:
    """
    Get all dataset files for the given configuration.
    
    Args:
        config: Dataset loader configuration
        
    Returns:
        List of dataset file paths
    """
    pattern = f"{config.language}_{config.dataset_type}_*.txt"
    files = sorted(config.dataset_path.glob(pattern))
    
    if not files:
        raise ValueError(f"No data files found at {config.dataset_path} matching {pattern}")
    
    # If using sharding for distributed training, select only appropriate files
    if config.shard_id is not None and config.num_shards is not None:
        files = [files[i] for i in range(len(files)) if i % config.num_shards == config.shard_id]
    
    return files


def cache_file_path(config: DatasetLoaderConfig, data_file: Path) -> Optional[Path]:
    """
    Generate cache file path for processed features.
    
    Args:
        config: Dataset loader configuration
        data_file: Data file path
        
    Returns:
        Cache file path or None if caching is disabled
    """
    if not config.cache_dir:
        return None
        
    # Create a unique cache filename based on settings
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract the file suffix (e.g., "0" from "python_train_0.txt")
    file_suffix = data_file.stem.split("_")[-1]
    
    cache_name = (f"{config.language}_{config.dataset_type}_{file_suffix}_"
                 f"{config.max_seq_length}_{config.reranker_type}.pt")
    
    return cache_dir / cache_name


def load_and_cache_file(
    config: DatasetLoaderConfig,
    data_file: Path
) -> TensorDataset:
    """
    Load and process a single data file.
    
    Args:
        config: Dataset loader configuration
        data_file: Data file path
        
    Returns:
        TensorDataset containing the processed features
    """
    # Check if we have a cached version
    cache_path = cache_file_path(config, data_file)
    if cache_path and cache_path.exists():
        logger.info(f"Loading cached features from {cache_path}")
        return torch.load(cache_path)
    
    # Process the file
    logger.info(f"Processing data file {data_file}")
    processor = RerankerProcessor()
    examples = processor._create_examples(
        processor._read_tsv(str(data_file)), 
        set_type=config.dataset_type
    )
    
    logger.info(f"Converting {len(examples)} examples to features")
    features = convert_examples_to_features(
        examples,
        config.max_seq_length,
        config.tokenizer,
        config.reranker_type,
        cls_token=config.tokenizer.cls_token,
        sep_token=config.tokenizer.sep_token,
        pad_token=config.tokenizer.pad_token_id,
        cls_token_segment_id=0,
        pad_token_segment_id=0
    )
    
    # Convert to tensor dataset
    if config.reranker_type == "pointwise":
        # Input IDs, attention mask, token type IDs, labels, (optional) query IDs
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        
        if hasattr(features[0], 'query_id') and features[0].query_id is not None:
            all_query_ids = torch.tensor([f.query_id for f in features], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_query_ids)
        else:
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
            
    else:  # pairwise
        # Positive examples
        all_pos_input_ids = torch.tensor([f.pos_input_ids for f in features], dtype=torch.long)
        all_pos_attention_mask = torch.tensor([f.pos_attention_mask for f in features], dtype=torch.long)
        all_pos_token_type_ids = torch.tensor([f.pos_token_type_ids for f in features], dtype=torch.long)
        
        # Negative examples
        all_neg_input_ids = torch.tensor([f.neg_input_ids for f in features], dtype=torch.long)
        all_neg_attention_mask = torch.tensor([f.neg_attention_mask for f in features], dtype=torch.long)
        all_neg_token_type_ids = torch.tensor([f.neg_token_type_ids for f in features], dtype=torch.long)
        
        # Create dummy labels (all 1s since pos should be ranked higher than neg)
        all_labels = torch.ones(len(features), dtype=torch.long)
        
        dataset = TensorDataset(
            all_pos_input_ids, all_pos_attention_mask, all_pos_token_type_ids,
            all_neg_input_ids, all_neg_attention_mask, all_neg_token_type_ids,
            all_labels
        )
    
    # Cache the dataset if caching is enabled
    if cache_path:
        logger.info(f"Caching features to {cache_path}")
        torch.save(dataset, cache_path)
    
    return dataset


def load_datasets(config: DatasetLoaderConfig) -> Union[TensorDataset, ConcatDataset]:
    """
    Load and process all data files for the given configuration.
    
    Args:
        config: Dataset loader configuration
        
    Returns:
        TensorDataset or ConcatDataset containing all processed features
    """
    # Get all data files
    data_files = get_dataset_files(config)
    logger.info(f"Found {len(data_files)} data files for {config.language}/{config.dataset_type}")
    
    # Process each file in parallel if multiple files exist
    datasets = []
    
    if len(data_files) > 1 and config.num_workers > 1:
        logger.info(f"Loading datasets in parallel with {config.num_workers} workers")
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.num_workers) as executor:
            # Create a dict mapping futures to their file description for better progress reporting
            future_to_file = {
                executor.submit(load_and_cache_file, config, data_file): data_file 
                for data_file in data_files
            }
            
            # Process results as they complete
            for future in tqdm(
                concurrent.futures.as_completed(future_to_file), 
                total=len(data_files),
                desc=f"Loading {config.dataset_type} data"
            ):
                data_file = future_to_file[future]
                try:
                    dataset = future.result()
                    datasets.append(dataset)
                    logger.info(f"Successfully loaded dataset from {data_file}")
                except Exception as e:
                    logger.error(f"Error processing {data_file}: {str(e)}")
                    raise
    else:
        # Serial processing for single file or when only one worker is specified
        for data_file in tqdm(data_files, desc=f"Loading {config.dataset_type} data"):
            dataset = load_and_cache_file(config, data_file)
            datasets.append(dataset)
    
    # Combine all datasets
    if len(datasets) == 1:
        return datasets[0]
    else:
        return ConcatDataset(datasets)