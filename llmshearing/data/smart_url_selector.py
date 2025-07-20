#!/usr/bin/env python3
"""
Advanced URL sampler that estimates file sizes to minimize downloads.
This version tries to download just enough files to meet the token requirements.
"""

import json
import random
import numpy as np
import os
import argparse
import urllib.request
import tempfile
from tqdm import tqdm


def get_domain_from_url(url):
    """Extract domain from RedPajama URL."""
    parts = url.split('/')
    for i, part in enumerate(parts):
        if part == 'v1.0.0' and i + 1 < len(parts):
            return parts[i + 1]
    return None


def estimate_file_tokens(url, sample_ratio=0.1, seq_length=4096):
    """
    Estimate tokens in a file by downloading a small sample.
    Returns estimated total tokens in the file.
    """
    try:
        # Try to get file size from HTTP headers
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req, timeout=30) as response:
            file_size = int(response.headers.get('content-length', 0))
            
        if file_size > 0:
            # Rough estimation: JSONL files are ~70% text content
            # Average token is ~4 characters
            estimated_text_size = file_size * 0.7
            estimated_tokens = estimated_text_size / 4
            estimated_sequences = estimated_tokens / seq_length
            return int(estimated_sequences)
        
    except Exception as e:
        print(f"Could not estimate size for {url}: {e}")
    
    # Fallback: use domain-specific averages (rough estimates)
    domain = get_domain_from_url(url)
    domain_avg_sequences = {
        "arxiv": 2000,     # Academic papers, longer texts
        "book": 5000,      # Books, very long texts
        "c4": 1000,        # Web pages, medium length
        "cc": 500,         # Common crawl, shorter texts
        "github": 800,     # Code files, variable length
        "stackexchange": 300,  # Q&A, shorter
        "wiki": 1500,      # Wikipedia articles, medium-long
    }
    
    return domain_avg_sequences.get(domain, 1000)  # Default fallback


def select_optimal_files(domain_urls, target_sequences, domain):
    """
    Select the optimal set of files to download to meet target sequence count.
    Uses estimated file sizes to minimize downloads.
    """
    print(f"  Estimating file sizes for {len(domain_urls)} URLs...")
    
    # Estimate sequences for each file
    file_estimates = []
    for url in tqdm(domain_urls[:50], desc="Sampling files for estimation"):  # Only sample first 50 for speed
        estimated_seqs = estimate_file_tokens(url)
        file_estimates.append((url, estimated_seqs))
    
    # If we sampled fewer files, extend estimates to all files
    if len(file_estimates) < len(domain_urls):
        avg_seqs = sum(est[1] for est in file_estimates) / len(file_estimates)
        for url in domain_urls[len(file_estimates):]:
            file_estimates.append((url, avg_seqs))
    
    # Sort by estimated sequences (descending) to prefer larger files
    file_estimates.sort(key=lambda x: x[1], reverse=True)
    
    # Greedily select files until we have enough sequences
    selected_files = []
    total_estimated_seqs = 0
    
    for url, estimated_seqs in file_estimates:
        selected_files.append(url)
        total_estimated_seqs += estimated_seqs
        
        if total_estimated_seqs >= target_sequences * 1.2:  # 20% buffer
            break
    
    print(f"  Selected {len(selected_files)} files (estimated {total_estimated_seqs} sequences for target {target_sequences})")
    return selected_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls_file", type=str, default="urls.txt", help="File containing RedPajama URLs")
    parser.add_argument("--output_file", type=str, default="selected_urls.txt", help="Output file with selected URLs")
    parser.add_argument("--eval_seq", type=int, default=2, help="How many sequences to sample for eval for each domain")
    parser.add_argument("--seq_length", type=int, default=4096, help="Sequence length")
    parser.add_argument("--for_prune", type=float, default=0.001, help="How many tokens (billion) sampled for pruning")
    parser.add_argument("--for_ft", type=float, default=0.001, help="How many tokens (billion) sampled for ft")
    
    args = parser.parse_args()

    # RedPajama domain sampling rates
    domain_ratios = {
        "arxiv": 0.025, 
        "book": 0.045, 
        "c4": 0.15,
        "cc": 0.67,
        "github": 0.045, 
        "stackexchange": 0.02, 
        "wiki": 0.045
    }

    # Calculate target sequences
    seq_1b = 1000000000 // args.seq_length
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

    print("URLs per domain:")
    for domain, urls_list in domain_to_urls.items():
        print(f"  {domain}: {len(urls_list)} files")

    # Select optimal files for each domain
    selected_urls = []
    random.seed(42)
    
    for domain, domain_urls in domain_to_urls.items():
        if not domain_urls:
            continue
            
        print(f"\nOptimizing file selection for domain: {domain}")
        
        # Calculate target sequences for this domain
        domain_eval_target = args.eval_seq
        domain_prune_target = int(for_prune_total * domain_ratios[domain])
        domain_ft_target = int(for_ft_total * domain_ratios[domain])
        total_target = domain_eval_target + domain_prune_target + domain_ft_target
        
        print(f"  Target: {total_target} sequences (eval: {domain_eval_target}, prune: {domain_prune_target}, ft: {domain_ft_target})")
        
        # Shuffle URLs for randomness
        random.shuffle(domain_urls)
        
        # Select optimal files
        if total_target > 0:
            selected_domain_urls = select_optimal_files(domain_urls, total_target, domain)
            selected_urls.extend(selected_domain_urls)
        else:
            print(f"  No sequences needed for domain {domain}")

    # Save selected URLs
    print(f"\nSaving {len(selected_urls)} selected URLs to {args.output_file}")
    with open(args.output_file, 'w') as f:
        for url in selected_urls:
            f.write(url + '\n')
    
    print("URL selection complete!")
    print(f"Selected {len(selected_urls)} files out of {len(urls)} total files")
    print(f"Estimated download reduction: {(1 - len(selected_urls)/len(urls))*100:.1f}%")


if __name__ == "__main__":
    main()
