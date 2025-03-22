import os
import json
import gzip
import argparse
import random
import glob
import concurrent.futures
from enum import Enum
from typing import Dict, List, Tuple, Set, Any, Optional, Iterator, Union
from pathlib import Path

from pydantic import BaseModel, Field, field_validator
from tqdm import tqdm


class DatasetType(str, Enum):
    """Dataset types in CodeSearchNet."""
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'


class CodeSearchNetConfig(BaseModel):
    """Configuration for CodeSearchNet dataset processing."""
    input_dir: Path = Field(..., description="Root directory of CodeSearchNet dataset")
    output_dir: Path = Field(..., description="Output directory for reranker format files")
    language: str = Field(..., description="Programming language to process")
    num_negatives: int = Field(5, description="Number of negative examples per positive example")
    seed: int = Field(42, description="Random seed for reproducibility")
    num_workers: int = Field(4, description="Number of worker threads")
    
    @field_validator('input_dir')
    def validate_input_dir(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Input directory '{v}' does not exist")
        return v
    
    @field_validator('output_dir')
    def validate_output_dir(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    def get_input_files(self, dataset_type: DatasetType) -> List[Path]:
        """Get all input files for a given dataset type."""
        pattern = f"{self.language}_{dataset_type.value}_*.jsonl.gz"
        path = self.input_dir / self.language / "final" / "jsonl" / dataset_type.value
        
        if not path.exists():
            raise ValueError(f"Path does not exist: {path}")
            
        return sorted(path.glob(pattern))
    
    def get_output_path(self, input_file: Path, dataset_type: DatasetType) -> Path:
        """Generate output path for a given input file."""
        # Extract the suffix number from the input filename
        # Example: python_train_0.jsonl.gz -> 0
        suffix = input_file.stem.split('_')[-1].replace('.jsonl', '')
        
        # Create the output directory structure
        output_subdir = self.output_dir / self.language / dataset_type.value
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        # Create the output filename
        output_filename = f"{self.language}_{dataset_type.value}_{suffix}.txt"
        return output_subdir / output_filename


class Example(BaseModel):
    """An example for reranker training."""
    label: int
    query_id: int
    url: str
    query: str
    code: str


def format_str(string: str) -> str:
    """
    Format string by replacing newlines with spaces.
    
    Args:
        string: Input string
        
    Returns:
        Formatted string
    """
    for char in ['\r\n', '\r', '\n']:
        string = string.replace(char, ' ')
    return string


def read_jsonl_gz(file_path: Path) -> List[Dict[str, Any]]:
    """
    Read data from a gzipped JSONL file.
    
    Args:
        file_path: Path to the input file
        
    Returns:
        List of dictionaries containing the data
    """
    data = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line in {file_path}")
                continue
    return data


def prepare_reranker_file(
    input_file: Path,
    output_file: Path,
    num_negatives: int = 5,
    seed: int = 42
) -> Tuple[int, int]:
    """
    Process a single input file into reranker format.
    
    Args:
        input_file: Path to the input JSONL.GZ file
        output_file: Path to the output TXT file
        num_negatives: Number of negative examples per positive example
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (number of examples created, number of unique queries)
    """
    # Set seed with the file index to ensure different random selections across files
    file_seed = seed + hash(str(input_file)) % 10000
    random.seed(file_seed)
    
    # Read input data
    data = read_jsonl_gz(input_file)
    if not data:
        print(f"Warning: No data found in {input_file}")
        return 0, 0
    
    # Build a dictionary of query_id -> examples
    query_dict: Dict[str, Dict[str, Any]] = {}
    for i, example in enumerate(data):
        query = ' '.join(example.get('docstring_tokens', []))
        if not query:  # Skip examples without queries
            continue
            
        code = ' '.join([format_str(token) for token in example.get('code_tokens', [])])
        url = example.get('url', '')
        
        if query not in query_dict:
            query_dict[query] = {
                'query_id': len(query_dict),
                'positive': [(i, url, code)],
                'all_indices': [i]
            }
        else:
            query_dict[query]['positive'].append((i, url, code))
            query_dict[query]['all_indices'].append(i)
    
    # Create training examples
    examples: List[Example] = []
    
    for query, query_data in query_dict.items():
        query_id = query_data['query_id']
        positive_examples: List[Tuple[int, str, str]] = query_data['positive']
        
        # Add positive examples (label=1)
        for idx, url, code in positive_examples:
            examples.append(Example(
                label=1,
                query_id=query_id,
                url=url,
                query=query,
                code=code
            ))
        
        # Add negative examples (label=0)
        # Sample indices that are not positive for this query
        all_indices: Set[int] = set(range(len(data)))
        positive_indices: Set[int] = set([idx for idx, _, _ in positive_examples])
        negative_candidates = list(all_indices - positive_indices)
        
        # Sample negative examples
        num_neg = min(num_negatives * len(positive_examples), len(negative_candidates))
        if num_neg > 0:  # Ensure we have negative examples to sample
            negative_indices = random.sample(negative_candidates, num_neg)
            
            for idx in negative_indices:
                example = data[idx]
                neg_code = ' '.join([format_str(token) for token in example.get('code_tokens', [])])
                neg_url = example.get('url', '')
                
                examples.append(Example(
                    label=0,
                    query_id=query_id,
                    url=neg_url,
                    query=query,
                    code=neg_code
                ))
    
    # Write examples to output file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in examples:
            line = f"{example.label}<CODESPLIT>{example.query_id}<CODESPLIT>{example.url}" \
                   f"<CODESPLIT>{example.query}<CODESPLIT>{example.code}\n"
            f.write(line)
    
    return len(examples), len(query_dict)


def process_dataset_type(
    config: CodeSearchNetConfig,
    dataset_type: DatasetType
) -> None:
    """
    Process all files for a specific dataset type.
    
    Args:
        config: Configuration
        dataset_type: Dataset type to process
    """
    input_files = config.get_input_files(dataset_type)
    if not input_files:
        print(f"No input files found for {dataset_type.value}")
        return
    
    print(f"Processing {len(input_files)} {dataset_type.value} files...")
    
    # Prepare arguments for parallel processing
    tasks = []
    for input_file in input_files:
        output_file = config.get_output_path(input_file, dataset_type)
        tasks.append((input_file, output_file, config.num_negatives, config.seed))
    
    # Process files in parallel
    total_examples = 0
    total_queries = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.num_workers) as executor:
        # Create a dict mapping futures to their file description for better progress reporting
        future_to_file = {
            executor.submit(prepare_reranker_file, *task): task[0] 
            for task in tasks
        }
        
        for future in tqdm(
            concurrent.futures.as_completed(future_to_file), 
            total=len(tasks),
            desc=f"Processing {dataset_type.value}"
        ):
            input_file = future_to_file[future]
            try:
                examples, queries = future.result()
                total_examples += examples
                total_queries += queries
                tqdm.write(f"Processed {input_file}: {examples} examples, {queries} queries")
            except Exception as e:
                tqdm.write(f"Error processing {input_file}: {str(e)}")
    
    print(f"Completed {dataset_type.value}: {total_examples} examples from {total_queries} unique queries")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Prepare data for CodeBERT reranker fine-tuning")
    parser.add_argument("--input_dir", required=True, type=str, 
                        help="Root directory of CodeSearchNet dataset")
    parser.add_argument("--output_dir", required=True, type=str, 
                        help="Output directory for reranker format files")
    parser.add_argument("--language", required=True, type=str,
                        help="Programming language to process (python, java, etc.)")
    parser.add_argument("--dataset_types", nargs='+', default=['train', 'valid', 'test'],
                        choices=['train', 'valid', 'test'], help="Dataset types to process")
    parser.add_argument("--num_negatives", type=int, default=5, 
                        help="Number of negative examples per positive example")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    parser.add_argument("--num_workers", type=int, default=12,
                        help="Number of worker threads")
    
    args = parser.parse_args()
    
    # Create config and validate
    config = CodeSearchNetConfig(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        language=args.language,
        num_negatives=args.num_negatives,
        seed=args.seed,
        num_workers=args.num_workers
    )
    
    # Process each requested dataset type
    for dataset_type_str in args.dataset_types:
        dataset_type = DatasetType(dataset_type_str)
        process_dataset_type(config, dataset_type)
    
    print(f"All requested dataset types processed successfully.")


if __name__ == "__main__":
    main()