import os
import json
import argparse
import random
from tqdm import tqdm

def format_str(string):
    """Format string by replacing newlines with spaces."""
    for char in ['\r\n', '\r', '\n']:
        string = string.replace(char, ' ')
    return string

def prepare_reranker_data(input_file, output_file, num_negatives=5, seed=42):
    """
    Prepare data for reranker fine-tuning from CodeSearchNet format.
    
    Args:
        input_file: Input file in jsonl format
        output_file: Output file in our reranker format
        num_negatives: Number of negative examples per positive example
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Read input data
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    # Build a dictionary of query_id -> examples
    query_dict = {}
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
    examples = []
    
    for query, query_data in tqdm(query_dict.items(), desc="Preparing data"):
        query_id = query_data['query_id']
        positive_examples = query_data['positive']
        
        # Add positive examples (label=1)
        for idx, url, code in positive_examples:
            examples.append({
                'label': 1,
                'query_id': query_id,
                'url': url,
                'query': query,
                'code': code
            })
        
        # Add negative examples (label=0)
        # Sample indices that are not positive for this query
        all_indices = set(range(len(data)))
        positive_indices = set([idx for idx, _, _ in positive_examples])
        negative_candidates = list(all_indices - positive_indices)
        
        # Sample negative examples
        num_neg = min(num_negatives * len(positive_examples), len(negative_candidates))
        negative_indices = random.sample(negative_candidates, num_neg)
        
        for idx in negative_indices:
            example = data[idx]
            neg_code = ' '.join([format_str(token) for token in example['code_tokens']])
            neg_url = example['url']
            
            examples.append({
                'label': 0,
                'query_id': query_id,
                'url': neg_url,
                'query': query,
                'code': neg_code
            })
    
    # Write examples to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in examples:
            line = f"{example['label']}<CODESPLIT>{example['query_id']}<CODESPLIT>{example['url']}" \
                   f"<CODESPLIT>{example['query']}<CODESPLIT>{example['code']}\n"
            f.write(line)
    
    print(f"Created {len(examples)} examples with {len(query_dict)} unique queries")

def main():
    parser = argparse.ArgumentParser(description="Prepare data for CodeBERT reranker fine-tuning")
    parser.add_argument("--input_file", required=True, help="Input file in jsonl format")
    parser.add_argument("--output_file", required=True, help="Output file in reranker format")
    parser.add_argument("--num_negatives", type=int, default=5, 
                       help="Number of negative examples per positive example")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Prepare data
    prepare_reranker_data(args.input_file, args.output_file, args.num_negatives, args.seed)

if __name__ == "__main__":
    main()