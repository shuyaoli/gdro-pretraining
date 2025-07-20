#!/usr/bin/env python3
"""
Optimized data preprocessing that samples URLs first, then downloads and tokenizes only selected files.
This avoids downloading the entire 1.3T RedPajama dataset when only ~50B tokens are needed.
"""

import json
import random
import numpy as np
from streaming import MDSWriter
import os
from tqdm import tqdm
import argparse
import urllib.request
import tempfile
from transformers import AutoTokenizer
from llama_tokenizer import Tokenizer


def make_dir_if_not_ex(path):
    if not os.path.exists(path):
        print("Make target folder:", path)
        os.makedirs(path)


def download_file(url, target_path):
    """Download a file from URL to target path."""
    print(f"Downloading {url} to {target_path}")
    try:
        urllib.request.urlretrieve(url, target_path)
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False


def get_domain_from_url(url):
    """Extract domain from RedPajama URL."""
    # Extract domain from URL path like: /redpajama-data-1T/v1.0.0/arxiv/file.jsonl
    parts = url.split('/')
    for i, part in enumerate(parts):
        if part == 'v1.0.0' and i + 1 < len(parts):
            return parts[i + 1]
    return None


def tokenize_file_content(file_path, tokenizer, seq_length=4096):
    """Tokenize a JSONL file and return sequences."""
    print(f"Tokenizing {file_path}...")
    lines = open(file_path).readlines()
    
    buffer = []
    data = []
    for line in tqdm(lines, desc="Tokenizing"):
        try:
            item = json.loads(line)
            tokens = buffer + tokenizer.encode(item["text"], bos=True, eos=True)
            buffer = []
            for start_id in range(0, len(tokens), seq_length):
                if start_id + seq_length < len(tokens):
                    data.append(tokens[start_id:start_id+seq_length])
                else:
                    buffer = tokens[start_id:]
                    break
        except json.JSONDecodeError:
            continue
    
    if data:
        return np.array(np.stack(data), dtype=np.uint16)
    else:
        return np.array([], dtype=np.uint16).reshape(0, seq_length)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls_file", type=str, default="urls.txt", help="File containing RedPajama URLs")
    parser.add_argument("--target_dir", type=str, default="mds_optimized_redpajama", help="Target directory to save MDS data")
    parser.add_argument("--temp_dir", type=str, default="temp_downloads", help="Temporary directory for downloads")
    parser.add_argument("--eval_seq", type=int, default=2, help="How many sequences to sample for eval for each domain")
    parser.add_argument("--seq_length", type=int, default=4096, help="Sequence length")
    parser.add_argument("--for_prune", type=float, default=0.001, help="How many tokens (billion) sampled for pruning")
    parser.add_argument("--for_ft", type=float, default=0.001, help="How many tokens (billion) sampled for ft")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer.model", help="Path to tokenizer")
    
    args = parser.parse_args()

    # RedPajama domain sampling rates
    domain_ratios = {
        "arxiv": 0.025, 
        "book": 0.045, 
        "c4": 0.15,  # Note: c4 instead of c4-rp
        "cc": 0.67,
        "github": 0.045, 
        "stackexchange": 0.02, 
        "wiki": 0.045
    }

    # Calculate how many sequences we need for each purpose
    seq_1b = 1000000000 // args.seq_length  # sequences for roughly 1B tokens
    for_prune_total = int(seq_1b * args.for_prune)
    for_ft_total = int(seq_1b * args.for_ft)

    print(f"Target sequences: eval={args.eval_seq * len(domain_ratios)}, prune={for_prune_total}, ft={for_ft_total}")

    # Read URLs and group by domain
    print("Reading URLs and grouping by domain...")
    with open(args.urls_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    domain_to_urls = {domain: [] for domain in domain_ratios}
    for url in urls:
        domain = get_domain_from_url(url)
        if domain in domain_to_urls:
            domain_to_urls[domain].append(url)
        elif domain == "c4":  # Handle c4 vs c4-rp naming
            if "c4" in domain_to_urls:
                domain_to_urls["c4"].append(url)

    print("URLs per domain:")
    for domain, urls_list in domain_to_urls.items():
        print(f"  {domain}: {len(urls_list)} files")

    # Create target directories
    make_dir_if_not_ex(args.target_dir)
    make_dir_if_not_ex(args.temp_dir)
    make_dir_if_not_ex(os.path.join(args.target_dir, "eval"))
    make_dir_if_not_ex(os.path.join(args.target_dir, "for_prune"))
    make_dir_if_not_ex(os.path.join(args.target_dir, "for_ft"))

    # Load tokenizer
    print("Loading tokenizer...")
    if os.path.exists(args.tokenizer_path):
        tokenizer = Tokenizer(args.tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    print("Tokenizer loaded.")

    random.seed(42)
    np.random.seed(42)

    # Process each domain
    for domain, domain_urls in domain_to_urls.items():
        if not domain_urls:
            print(f"No URLs found for domain {domain}, skipping...")
            continue
            
        print(f"\nProcessing domain: {domain}")
        
        # Calculate target sequences for this domain
        domain_eval_target = args.eval_seq
        domain_prune_target = int(for_prune_total * domain_ratios[domain])
        domain_ft_target = int(for_ft_total * domain_ratios[domain])
        
        total_sequences_needed = domain_eval_target + domain_prune_target + domain_ft_target
        print(f"  Need {total_sequences_needed} sequences (eval: {domain_eval_target}, prune: {domain_prune_target}, ft: {domain_ft_target})")
        
        # Estimate how many files we need to download
        # Assume average file has ~1000 sequences (this is a rough estimate)
        estimated_seqs_per_file = 1000
        files_needed = max(1, (total_sequences_needed // estimated_seqs_per_file) + 1)
        files_needed = min(files_needed, len(domain_urls))
        
        print(f"  Estimated files needed: {files_needed} out of {len(domain_urls)}")
        
        # Randomly sample files to download
        random.shuffle(domain_urls)
        selected_urls = domain_urls[:files_needed]
        
        # Create domain directories
        make_dir_if_not_ex(os.path.join(args.target_dir, "eval", domain))
        make_dir_if_not_ex(os.path.join(args.target_dir, "for_prune", domain))
        make_dir_if_not_ex(os.path.join(args.target_dir, "for_ft", domain))
        
        # Initialize MDS writers
        eval_writer = MDSWriter(
            columns={"tokens": "bytes", "set": "str"}, 
            out=os.path.join(args.target_dir, "eval", domain), 
            compression=None
        )
        prune_writer = MDSWriter(
            columns={"tokens": "bytes", "set": "str"}, 
            out=os.path.join(args.target_dir, "for_prune", domain), 
            compression=None
        )
        ft_writer = MDSWriter(
            columns={"tokens": "bytes", "set": "str"}, 
            out=os.path.join(args.target_dir, "for_ft", domain), 
            compression=None
        )
        
        eval_count = 0
        prune_count = 0
        ft_count = 0
        
        # Process files until we have enough sequences
        for i, url in enumerate(selected_urls):
            if eval_count >= domain_eval_target and prune_count >= domain_prune_target and ft_count >= domain_ft_target:
                print(f"  Reached targets for domain {domain}")
                break
                
            # Download file
            filename = os.path.basename(url)
            temp_file_path = os.path.join(args.temp_dir, filename)
            
            if not download_file(url, temp_file_path):
                continue
                
            try:
                # Tokenize file
                sequences = tokenize_file_content(temp_file_path, tokenizer, args.seq_length)
                
                if len(sequences) == 0:
                    print(f"  No sequences extracted from {filename}")
                    continue
                
                print(f"  Extracted {len(sequences)} sequences from {filename}")
                
                # Split sequences for different purposes
                indices = np.arange(len(sequences))
                np.random.shuffle(indices)
                
                # Allocate sequences to eval, prune, and ft
                allocated = 0
                
                # Eval first (highest priority)
                if eval_count < domain_eval_target:
                    eval_needed = min(domain_eval_target - eval_count, len(sequences) - allocated)
                    eval_indices = indices[allocated:allocated + eval_needed]
                    for idx in eval_indices:
                        eval_writer.write({
                            "tokens": sequences[idx].tobytes(),
                            "set": domain
                        })
                    eval_count += eval_needed
                    allocated += eval_needed
                
                # Prune second
                if prune_count < domain_prune_target and allocated < len(sequences):
                    prune_needed = min(domain_prune_target - prune_count, len(sequences) - allocated)
                    prune_indices = indices[allocated:allocated + prune_needed]
                    for idx in prune_indices:
                        prune_writer.write({
                            "tokens": sequences[idx].tobytes(),
                            "set": domain
                        })
                    prune_count += prune_needed
                    allocated += prune_needed
                
                # FT last
                if ft_count < domain_ft_target and allocated < len(sequences):
                    ft_needed = min(domain_ft_target - ft_count, len(sequences) - allocated)
                    ft_indices = indices[allocated:allocated + ft_needed]
                    for idx in ft_indices:
                        ft_writer.write({
                            "tokens": sequences[idx].tobytes(),
                            "set": domain
                        })
                    ft_count += ft_needed
                    allocated += ft_needed
                
                print(f"  Progress - eval: {eval_count}/{domain_eval_target}, prune: {prune_count}/{domain_prune_target}, ft: {ft_count}/{domain_ft_target}")
                
            except Exception as e:
                print(f"  Error processing {filename}: {e}")
            finally:
                # Clean up downloaded file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
        
        # Finish writers
        eval_writer.finish()
        prune_writer.finish()
        ft_writer.finish()
        
        print(f"  Completed domain {domain} - eval: {eval_count}, prune: {prune_count}, ft: {ft_count}")

    print("\nOptimized preprocessing complete!")
    print("Next step: Create eval_merge using merge_data.py")


if __name__ == "__main__":
    main()
