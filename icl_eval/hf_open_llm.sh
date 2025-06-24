#!/bin/bash

bsz="${bsz:-16}"
resultpath="${4:-tmp}"
echo $resultpath

# harness_dir="$HOME/LLM-Shearing/icl_eval"

cmd="python3 main.py --model=hf --model_args="pretrained=$1,dtype=float16" --tasks=$2 --num_fewshot=$3 --batch_size=$bsz --output_path=$harness_dir/result/$resultpath"
if [[ -n $5 ]]; then cmd="$cmd --limit=$5"; fi

$cmd 
