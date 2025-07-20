#!/bin/bash
# Optimized RedPajama preprocessing script
# This script samples URLs first, then downloads and tokenizes only selected files

set -e

# Configuration
URLS_FILE="../urls.txt"
TARGET_DIR="mds_optimized_redpajama"
TEMP_DIR="temp_downloads"
TOKENIZER_PATH="tokenizer.model"

# Sampling parameters (adjust as needed)
EVAL_SEQ=2            # sequences per domain for evaluation
FOR_PRUNE=0.001       # billion tokens for pruning (1M tokens)
FOR_FT=0.001         # billion tokens for fine-tuning (1M tokens)
SEQ_LENGTH=4096      # sequence length

echo "Starting optimized RedPajama preprocessing..."
echo "Configuration:"
echo "  URLs file: $URLS_FILE"
echo "  Target dir: $TARGET_DIR"
echo "  Temp dir: $TEMP_DIR"
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

# Create directories
mkdir -p "$TEMP_DIR"

# Run the optimized preprocessing
echo "Running optimized preprocessing..."
python optimized_sample_and_download.py \
    --urls_file "$URLS_FILE" \
    --target_dir "$TARGET_DIR" \
    --temp_dir "$TEMP_DIR" \
    --eval_seq "$EVAL_SEQ" \
    --seq_length "$SEQ_LENGTH" \
    --for_prune "$FOR_PRUNE" \
    --for_ft "$FOR_FT" \
    --tokenizer_path "$TOKENIZER_PATH"

# Create eval_merge (combines all domains for evaluation)
echo ""
echo "Creating eval_merge split..."
python3 -m llmshearing.data.merge_data \
    --input_dir "$TARGET_DIR/eval" \
    --output_dir "$TARGET_DIR/eval" \
    --output_split eval_merge \
    --split_names arxiv book c4 cc github stackexchange wiki

echo ""
echo "Optimized preprocessing complete!"
echo "Data structure:"
echo "  $TARGET_DIR/"
echo "    eval/"
echo "      arxiv/"
echo "      book/"
echo "      c4/"
echo "      cc/"
echo "      github/"
echo "      stackexchange/"
echo "      wiki/"
echo "      eval_merge/"
echo "    for_prune/"
echo "      [same domains]"
echo "    for_ft/"
echo "      [same domains]"
echo ""
echo "You can now use this data for training with the existing LLM shearing pipeline."

# Clean up temp directory
echo "Cleaning up temporary files..."
rm -rf "$TEMP_DIR"
echo "Done!"
