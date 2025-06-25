#!/bin/bash
MODEL_PATH=/mnt/disks/gdro-model-storage/
# composer_model_path, output_path, *other_args = sys.argv[2:]
COMPOSER_MODEL_PATH=$MODEL_PATH/doremi_ft1000ba_Adam/ep0-ba450-rank0.pt
OUTPUT_PATH=/mnt/disks/model_to_eval/doremi_ft1000ba_Adam


MODEL_CLASS=LlamaForCausalLM
HIDDEN_SIZE=2048
NUM_ATTENTION_HEADS=16
NUM_HIDDEN_LAYERS=24
INTERMEDIATE_SIZE=5504
MODEL_NAME=Sheared-Llama-1.3B


# INPUT_PATH=$HOME/LLM-Shearing/models/LLaMA-1-3-B-Pruned/state_dict.pt
mkdir -p $(dirname $OUTPUT_PATH)
python3 -m llmshearing.utils.composer_to_hf save_composer_to_hf $COMPOSER_MODEL_PATH $OUTPUT_PATH \
        model_class=${MODEL_CLASS} \
        hidden_size=${HIDDEN_SIZE} \
        num_attention_heads=${NUM_ATTENTION_HEADS} \
        num_hidden_layers=${NUM_HIDDEN_LAYERS} \
        intermediate_size=${INTERMEDIATE_SIZE} \
        num_key_value_heads=${NUM_ATTENTION_HEADS} \
        _name_or_path=${MODEL_NAME}
