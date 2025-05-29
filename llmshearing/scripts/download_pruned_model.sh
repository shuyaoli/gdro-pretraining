HF_MODEL_NAME=princeton-nlp/Sheared-LLaMA-1.3B-Pruned
OUTPUT_PATH=$HOME/LLM-Shearing/models/LLaMA-1-3-B-Pruned/state_dict.pt
mkdir -p $(dirname $OUTPUT_PATH)
python3 -m llmshearing.utils.composer_to_hf save_hf_to_composer $HF_MODEL_NAME $OUTPUT_PATH
