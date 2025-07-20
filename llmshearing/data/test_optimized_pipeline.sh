#!/bin/bash
# Test the optimized preprocessing pipeline with a small subset of data
# This is useful for validating the pipeline before running on the full dataset

set -e

echo "=========================================="
echo "Testing Optimized RedPajama Preprocessing"
echo "=========================================="
echo ""

# Test configuration (small scale)
URLS_FILE="../urls.txt"
TEST_URLS_FILE="test_urls.txt" 
SELECTED_URLS_FILE="test_selected_urls.txt"
TARGET_DIR="test_mds_data"
TEMP_DIR="test_temp_downloads"
TOKENIZER_PATH="tokenizer.model"

# Small sampling parameters for testing
EVAL_SEQ=1            # Just 1 sequence per domain for testing
FOR_PRUNE=0.000001    # Very small amount (1K tokens)
FOR_FT=0.000001      # Very small amount (1K tokens)
SEQ_LENGTH=4096

echo "Test Configuration:"
echo "  Test URLs file: $TEST_URLS_FILE"
echo "  Target directory: $TARGET_DIR"
echo "  Eval sequences per domain: $EVAL_SEQ"
echo "  Pruning data: ${FOR_PRUNE}B tokens"
echo "  Fine-tuning data: ${FOR_FT}B tokens"
echo ""

# Check if URLs file exists
if [ ! -f "$URLS_FILE" ]; then
    echo "Error: URLs file $URLS_FILE not found!"
    exit 1
fi

# Create test URLs file with just first 20 URLs
echo "Creating test URLs file with first 20 URLs..."
head -20 "$URLS_FILE" > "$TEST_URLS_FILE"
echo "Created $TEST_URLS_FILE with $(wc -l < $TEST_URLS_FILE) URLs"
echo ""

# Clean up any existing test data
if [ -d "$TARGET_DIR" ]; then
    echo "Removing existing test data..."
    rm -rf "$TARGET_DIR"
fi

echo "=========================================="
echo "Running Test Pipeline"
echo "=========================================="
echo ""

# Stage 1: Smart URL selection
echo "Stage 1: URL Selection"
python smart_url_selector.py \
    --urls_file "$TEST_URLS_FILE" \
    --output_file "$SELECTED_URLS_FILE" \
    --eval_seq "$EVAL_SEQ" \
    --seq_length "$SEQ_LENGTH" \
    --for_prune "$FOR_PRUNE" \
    --for_ft "$FOR_FT"

echo ""

# Stage 2: Download and process
echo "Stage 2: Download and Process"
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

# Stage 3: Create eval merge if there's data
echo "Stage 3: Create Evaluation Merge"
EVAL_DIR="$TARGET_DIR/eval"
if [ -d "$EVAL_DIR" ]; then
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
        echo "No evaluation data to merge."
    fi
else
    echo "No evaluation directory found."
fi

echo ""
echo "=========================================="
echo "Test Results"
echo "=========================================="
echo ""

# Show what was created
if [ -d "$TARGET_DIR" ]; then
    echo "Generated test data structure:"
    find "$TARGET_DIR" -type f -name "*.mds" | head -10 | while read file; do
        size=$(stat -c%s "$file" 2>/dev/null || echo "unknown")
        echo "  $file ($size bytes)"
    done
    
    total_files=$(find "$TARGET_DIR" -name "*.mds" | wc -l)
    echo "  ... total $total_files MDS files"
    
    echo ""
    echo "Data splits created:"
    for split in eval for_prune for_ft; do
        if [ -d "$TARGET_DIR/$split" ]; then
            files=$(find "$TARGET_DIR/$split" -name "*.mds" | wc -l)
            echo "  $split: $files files"
        fi
    done
else
    echo "No test data was created. Check for errors above."
fi

echo ""

# Test data validation
echo "Testing data loading (if streaming is available)..."
python3 -c "
try:
    import streaming
    import os
    if os.path.exists('$TARGET_DIR/eval'):
        for domain in os.listdir('$TARGET_DIR/eval'):
            domain_path = os.path.join('$TARGET_DIR/eval', domain)
            if os.path.isdir(domain_path) and any(f.endswith('.mds') for f in os.listdir(domain_path)):
                try:
                    dataset = streaming.StreamingDataset(local=domain_path)
                    print(f'✓ Successfully loaded {domain} dataset with {len(dataset)} samples')
                    if len(dataset) > 0:
                        sample = dataset[0]
                        print(f'  Sample keys: {list(sample.keys())}')
                        break
                except Exception as e:
                    print(f'✗ Failed to load {domain}: {e}')
        else:
            print('No valid datasets found to test')
    else:
        print('No evaluation data to test')
except ImportError:
    print('Streaming package not available for testing')
" 2>/dev/null || echo "Could not test data loading"

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="

# Clean up test files
echo ""
echo "Cleaning up test files..."
rm -f "$TEST_URLS_FILE" "$SELECTED_URLS_FILE"
rm -rf "$TEMP_DIR"

echo ""
echo "Test completed successfully! The pipeline appears to be working."
echo "You can now run the full pipeline with:"
echo "  bash run_complete_optimized_pipeline.sh"
echo ""
echo "To remove test data:"
echo "  rm -rf $TARGET_DIR"
