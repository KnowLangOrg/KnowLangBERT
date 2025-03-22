import os
import json
import gzip
import argparse
import random
from typing import Dict, List, Tuple, Set, Any

from pydantic import BaseModel, Field, field_validator
from tqdm import tqdm


class DataPrepConfig(BaseModel):
    """Configuration for data preparation."""
    input_file: str = Field(..., description="Input file in jsonl or jsonl.gz format")
    output_file: str = Field(..., description="Output file in reranker format")
    num_negatives: int = Field(5, description="Number of negative examples per positive example")
    seed: int = Field(42, description="Random seed for reproducibility")
    
    @field_validator('input_file')
    def validate_input_file(cls, v: str) -> str:
        if not os.path.exists(v):
            raise ValueError(f"Input file '{v}' does not exist")
        return v
    
    @field_validator('output_file')
    def validate_output_dir(cls, v: str) -> str:
        output_dir = os.path.dirname(v)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        return v


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


def read_jsonl_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Read data from a JSONL file, which may be gzipped.
    
    Args:
        file_path: Path to the input file
        
    Returns:
        List of dictionaries containing the data
    """
    data = []
    
    # Check if the file is gzipped
    is_gzipped = file_path.endswith('.gz')
    
    # Read directly from gzipped or normal file
    open_func = gzip.open if is_gzipped else open
    mode = 'rt' if is_gzipped else 'r'
    
    with open_func(file_path, mode, encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    return data


def prepare_reranker_data(
    input_file: str, 
    output_file: str, 
    num_negatives: int = 5, 
    seed: int = 42
) -> None:
    """
    Prepare data for reranker fine-tuning from CodeSearchNet format.
    
    Args:
        input_file: Input file in jsonl or jsonl.gz format
        output_file: Output file in our reranker format
        num_negatives: Number of negative examples per positive example
        seed: Random seed for reproducibility
    """
    # Validate configuration
    config = DataPrepConfig(
        input_file=input_file,
        output_file=output_file,
        num_negatives=num_negatives,
        seed=seed
    )
    
    random.seed(config.seed)
    
    # Read input data
    data = read_jsonl_data(config.input_file)
    
    # Build a dictionary of query_id -> examples
    query_dict: Dict[str, Dict[str, Any]] = {}
    for i, example in enumerate(data):
        query = ' '.join(example['docstring_tokens'])
        code = ' '.join([format_str(token) for token in example['code_tokens']])
        url = example['url']
        
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
    
    for query, query_data in tqdm(query_dict.items(), desc="Preparing data"):
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
        num_neg = min(config.num_negatives * len(positive_examples), len(negative_candidates))
        negative_indices = random.sample(negative_candidates, num_neg)
        
        for idx in negative_indices:
            example = data[idx]
            neg_code = ' '.join([format_str(token) for token in example['code_tokens']])
            neg_url = example['url']
            
            examples.append(Example(
                label=0,
                query_id=query_id,
                url=neg_url,
                query=query,
                code=neg_code
            ))
    
    # Write examples to output file
    with open(config.output_file, 'w', encoding='utf-8') as f:
        for example in examples:
            line = f"{example.label}<CODESPLIT>{example.query_id}<CODESPLIT>{example.url}" \
                   f"<CODESPLIT>{example.query}<CODESPLIT>{example.code}\n"
            f.write(line)
    
    print(f"Created {len(examples)} examples with {len(query_dict)} unique queries")

def main():
    parser = argparse.ArgumentParser(description="Prepare data for CodeBERT reranker fine-tuning")
    parser.add_argument("--input_file", required=True, help="Input file in jsonl or jsonl.gz format")
    parser.add_argument("--output_file", required=True, help="Output file in reranker format")
    parser.add_argument("--num_negatives", type=int, default=5, 
                       help="Number of negative examples per positive example")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Prepare data
    prepare_reranker_data(
        args.input_file, 
        args.output_file, 
        args.num_negatives, 
        args.seed,
    )

if __name__ == "__main__":
    main()