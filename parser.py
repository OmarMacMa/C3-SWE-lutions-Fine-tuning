"""Converts a SWE-bench-like dataset into a training dataset for a the fine tuning declaration in desired_json.json format."""

import re
import pandas as pd
import json
from typing import List, Dict, Any, Tuple

def parse_git_diff(diff_text: str) -> Tuple[str, str, str]:
    """
    Parse a git diff to extract the original and modified text, and filepath.
    
    Args:
        diff_text (str): The git diff content
        
    Returns:
        tuple: (original_code, new_code, filepath)
    """
    # Handle case where diff_text might be None or NaN
    if pd.isna(diff_text) or diff_text is None or not isinstance(diff_text, str):
        return "", "", ""
    
    lines = diff_text.splitlines()
    original_file = []
    new_file = []
    filepath = ""
    
    # Extract filepath from the diff header
    for line in lines:
        if line.startswith('+++'):
            match = re.match(r'\+\+\+ [ab]/(.+)', line)
            if match:
                filepath = match.group(1)
            else:
                filepath_parts = line.split(' ', 1)
                if len(filepath_parts) > 1:
                    filepath = filepath_parts[1].strip()
            break
    
    # If we didn't find a +++ line, try the diff --git line
    if not filepath:
        for line in lines:
            if line.startswith('diff --git'):
                match = re.match(r'diff --git a/(.+) b/(.+)', line)
                if match:
                    filepath = match.group(2)
                break
    
    # Skip header lines
    is_content = False
    for line in lines:
        # Start processing after the @@ line
        if re.match(r'^@@\s[-+]', line):
            is_content = True
            continue
            
        if not is_content:
            continue
            
        # Process content lines
        if line.startswith('-') and not line.startswith('---'):
            original_file.append(line[1:])
        elif line.startswith('+') and not line.startswith('+++'):
            new_file.append(line[1:])
        # Lines without +/- (context lines) appear in both versions
        elif not line.startswith('+') and not line.startswith('-'):
            original_file.append(line)
            new_file.append(line)
    
    return '\n'.join(original_file), '\n'.join(new_file), filepath

def create_training_examples(parquet_file: str, 
                            patch: str = 'patch',
                            problem_column: str = 'problem_statement') -> List[Dict[str, Any]]:
    """
    Create training examples in the required JSON format from a parquet file.
    
    Args:
        parquet_file (str): Path to parquet file
        patch (str): Column name containing the diff text
        problem_column (str): Column name containing the problem statement
        
    Returns:
        List[Dict]: List of training examples in the required format
    """
    df = pd.read_parquet(parquet_file)
    
    # Ensure required columns exist
    if patch not in df.columns:
        raise ValueError(f"Column '{patch}' not found in parquet file")
    if problem_column not in df.columns:
        raise ValueError(f"Column '{problem_column}' not found in parquet file")
    
    training_examples = []
    
    # Process each row in the dataframe
    for _, row in df.iterrows():
        # Parse the diff
        original_code, new_code, filepath = parse_git_diff(row[patch])
        
        # Skip if any required field is empty
        if not original_code or not new_code or not filepath:
            continue
        
        # Create the training example
        example = {
            "instruction": row[problem_column],
            "input": original_code,
            "output": new_code
            # "instruction": row[problem_column],
            # "input": {
            #     "code_snippet": original_code,
            #     "file_path": filepath
            # },
            # "output": new_code
        }

        training_examples.append(example)

    return training_examples

def save_training_examples(examples: List[Dict[str, Any]], output_file: str, format_type: str = 'json'):
    """
    Save training examples to a file.
    
    Args:
        examples (List[Dict]): List of training examples
        output_file (str): Path to output file
        format_type (str): 'jsonl' for one JSON object per line, 'json' for a single JSON array
    """
    if format_type.lower() == 'jsonl':
        # JSONL format (one JSON object per line)
        with open(output_file, 'w') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
    else:
        # Single JSON array format
        with open(output_file, 'w') as f:
            json.dump(examples, f, indent=2)

# Example usage
def process_and_save_examples(parquet_file: str, output_file: str, format_type: str = 'json'):
    """
    Process parquet file and save training examples.
    
    Args:
        parquet_file (str): Path to parquet file
        output_file (str): Path to output file
        format_type (str): 'jsonl' for one JSON object per line, 'json' for a single JSON array
    """
    examples = create_training_examples(parquet_file)
    save_training_examples(examples, output_file, format_type)
    print(f"Created {len(examples)} training examples and saved to {output_file} in {format_type} format")


if __name__ == "__main__":
    parquet_file = "./SWE-bench/data/dev-00000-of-00001.parquet"
    output_file = "training_dataset_.json"  # .json extension for the array format
    process_and_save_examples(parquet_file, output_file, format_type='json')