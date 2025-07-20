#!/bin/bash
# Complete optimized RedPajama preprocessing pipeline
# This script implements a 3-stage approach:
# 1. Smart URL selection (estimates file sizes to minimize downloads)
# 2. Download and tokenize only selected files  
# 3. Create MDS format for training

set -e

echo "=========================================="
echo "Optimized RedPajama Preprocessing Pipeline"
echo "=========================================="
echo ""

# Configuration
URLS_FILE="../urls.txt"
SELECTED_URLS_FILE="selected_urls.txt"
TARGET_DIR="mds_optimized_redpajama"
TEMP_DIR="temp_downloads"
TOKENIZER_PATH="tokenizer.model"

# Sampling parameters (adjust as needed)
EVAL_SEQ=2            # sequences per domain for evaluation
FOR_PRUNE=0.038       # billion tokens for pruning (38M tokens = ~50B tokens total including ft)
FOR_FT=0.012         # billion tokens for fine-tuning (12M tokens)
SEQ_LENGTH=4096      # sequence length

echo "Configuration:"
echo "  Original URLs file: $URLS_FILE"
echo "  Selected URLs file: $SELECTED_URLS_FILE"
echo "  Target directory: $TARGET_DIR"
echo "  Temp directory: $TEMP_DIR"
echo "  Eval sequences per domain: $EVAL_SEQ"
echo "  Pruning data: ${FOR_PRUNE}B tokens"
echo "  Fine-tuning data: ${FOR_FT}B tokens"
echo "  Sequence length: $SEQ_LENGTH"
echo ""

# Check if URLs file exists
if [ ! -f "$URLS_FILE" ]; then
    echo "Error: URLs file $URLS_FILE not found!"
    echo "Please make sure the urls.txt file is in the correct location."
    exit 1
fi

# Check if tokenizer exists
if [ ! -f "$TOKENIZER_PATH" ]; then
    echo "Warning: Tokenizer file $TOKENIZER_PATH not found!"
    echo "Will use HuggingFace tokenizer instead (slower)."
fi

echo "=========================================="
echo "Stage 1: Smart URL Selection"
echo "=========================================="
echo ""

# Stage 1: Smart URL selection
echo "Running smart URL selection to minimize downloads..."
python smart_url_selector.py \
    --urls_file "$URLS_FILE" \
    --output_file "$SELECTED_URLS_FILE" \
    --eval_seq "$EVAL_SEQ" \
    --seq_length "$SEQ_LENGTH" \
    --for_prune "$FOR_PRUNE" \
    --for_ft "$FOR_FT"

if [ ! -f "$SELECTED_URLS_FILE" ]; then
    echo "Error: URL selection failed!"
    exit 1
fi

TOTAL_URLS=$(wc -l < "$URLS_FILE")
SELECTED_URLS=$(wc -l < "$SELECTED_URLS_FILE")
REDUCTION=$(echo "scale=1; (1 - $SELECTED_URLS / $TOTAL_URLS) * 100" | bc -l)

echo ""
echo "URL Selection Results:"
echo "  Total URLs: $TOTAL_URLS"
echo "  Selected URLs: $SELECTED_URLS" 
echo "  Download reduction: $REDUCTION%"
echo ""

echo "=========================================="
echo "Stage 2: Download and Process Selected Files"
echo "=========================================="
echo ""

# Stage 2: Download and process selected files
echo "Downloading and processing selected files..."
python optimized_sample_and_download.py \
    --urls_file "$SELECTED_URLS_FILE" \
    --target_dir "$TARGET_DIR" \
    --temp_dir "$TEMP_DIR" \
    --eval_seq "$EVAL_SEQ" \
    --seq_length "$SEQ_LENGTH" \
    --for_prune "$FOR_PRUNE" \
    --for_ft "$FOR_FT" \
    --tokenizer_path "$TOKENIZER_PATH"

echo ""
echo "=========================================="
echo "Stage 3: Create Evaluation Merge"
echo "=========================================="
echo ""

# Stage 3: Create eval_merge (combines all domains for evaluation)
echo "Creating eval_merge split..."

# Check which domains actually have data
EVAL_DIR="$TARGET_DIR/eval"
DOMAINS=""
for domain in arxiv book c4 cc github stackexchange wiki; do
    if [ -d "$EVAL_DIR/$domain" ] && [ "$(ls -A $EVAL_DIR/$domain 2>/dev/null)" ]; then
        DOMAINS="$DOMAINS $domain"
    fi
done

if [ -n "$DOMAINS" ]; then
    echo "Merging domains:$DOMAINS"
    python3 -m llmshearing.data.merge_data \
        --input_dir "$EVAL_DIR" \
        --output_dir "$EVAL_DIR" \
        --output_split eval_merge \
        --split_names $DOMAINS
else
    echo "Warning: No evaluation data found to merge!"
fi

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""

# Display results
echo "Final data structure:"
echo "  $TARGET_DIR/"
if [ -d "$TARGET_DIR/eval" ]; then
    echo "    eval/"
    for domain in arxiv book c4 cc github stackexchange wiki; do
        if [ -d "$TARGET_DIR/eval/$domain" ]; then
            echo "      $domain/"
        fi
    done
    if [ -d "$TARGET_DIR/eval/eval_merge" ]; then
        echo "      eval_merge/"
    fi
fi

if [ -d "$TARGET_DIR/for_prune" ]; then
    echo "    for_prune/"
    for domain in arxiv book c4 cc github stackexchange wiki; do
        if [ -d "$TARGET_DIR/for_prune/$domain" ]; then
            echo "      $domain/"
        fi
    done
fi

if [ -d "$TARGET_DIR/for_ft" ]; then
    echo "    for_ft/"
    for domain in arxiv book c4 cc github stackexchange wiki; do
        if [ -d "$TARGET_DIR/for_ft/$domain" ]; then
            echo "      $domain/"
        fi
    done
fi

echo ""

# Show data statistics
echo "Data Statistics:"
for split in eval for_prune for_ft; do
    if [ -d "$TARGET_DIR/$split" ]; then
        total_files=0
        for domain_dir in "$TARGET_DIR/$split"/*; do
            if [ -d "$domain_dir" ]; then
                files=$(find "$domain_dir" -name "*.mds" 2>/dev/null | wc -l)
                total_files=$((total_files + files))
            fi
        done
        echo "  $split: $total_files MDS shard files"
    fi
done

echo ""
echo "You can now use this data for training with the existing LLM shearing pipeline."
echo "The data is compatible with the original preprocessing format."

# Clean up
echo ""
echo "Cleaning up temporary files..."
rm -rf "$TEMP_DIR"
rm -f "$SELECTED_URLS_FILE"
echo "Done!"
echo ""
echo "=========================================="
echo "Efficiency Summary"
echo "=========================================="
echo "  Original approach: Download 1.3T tokens, then sample ~50B"
echo "  Optimized approach: Download ~50B tokens directly"
echo "  Estimated speedup: 26x faster"
echo "  Estimated storage savings: 96% less disk space needed"
echo "=========================================="
