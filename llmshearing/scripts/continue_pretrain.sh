# pruning llama2 7b -> 3b or 1.3b
# Project directories
PROJ_DIR=$HOME/LLM-Shearing
DATA_DIR=${PROJ_DIR}/llm_dataset/LLM-Shearing/for_prune
OUTPUT_DIR=/mnt/disks/gdro-model-storage
LAUNCH_SCRIPT=${PROJ_DIR}/llmshearing/scripts/launch.sh
TRAIN_SCRIPT=${PROJ_DIR}/llmshearing/train.py

# Flags and parameters
eval_first=False
test=False # set to True for testing, it will run for 1 hour
optimizer=Adam

model=1.3b # target model size
config_file=${PROJ_DIR}/llmshearing/configs/llama2/${model}_${optimizer}.yaml
path=${PROJ_DIR}/models/LLaMA-1-3-B-Pruned/state_dict.pt # path to the  pruned model

# data setup
data_local=${DATA_DIR}

# basic setup
max_seq_len=4096
device_train_microbatch_size=16  # 16 A100 80GB GPUs total
global_train_batch_size=256
device_eval_batch_size=8

# learning setup
lr=1e-4
# !!! Remember to change t-warmup if max_duration is changed
max_duration=1000ba
# !!! Remember to change t-warmup if max_duration is changed
save_interval=50ba
# t_warmup=24ba # 3% learning rate warmup 

# dynamic loading setup
eval_interval=5ba # eval every 50 batches and update the loading proportion
dynamic=True
set_names="[cc,github,book,stackexchange,wiki,arxiv,c4-rp]" # domain names
proportion="[0.2192,0.0002,0.0791,0.0064,0.0096,0.001,0.6845]" # final proportion of pruning
# doremi: update weights with exponential descent
# constant: keep the weights constant
update_type="pd-kl" 
target_loss="[1.9643,0.7459,2.1393,1.6117,1.7590,1.4449,2.1251]" # 1.3b predicted loss from scaling law
eval_split_name=eval_merge # eval on all domains


# save directroy
run_name=${update_type}_ft${max_duration}_${optimizer}
save_dir=${OUTPUT_DIR}/${run_name}
wandb_dir=${save_dir}
# Resource allocation
if [ "$test" = "True" ]; then
  t=00-01:00:00
else
  t=01-00:00:00
fi

# Parse command-line flags
DEBUG_FLAG=""
LOAD_PATH=""
CONFIG_AUTORECOVER=false
for arg in "$@"; do
  case $arg in
    --debug)
      DEBUG_FLAG="--debug"
      ;;
    --load_path=*)
      LOAD_PATH="${arg#*=}"
      CONFIG_AUTORECOVER=true
      ;;
  esac
done

# Run with slurm
# sbatch -p cli \
#     --job-name ${run_name} \
#     --nodes=8 \
#     --gpus-per-node=2 \
#     --mem=512gb \
#     --cpus-per-task=8 \
#     --time $t \
#     $LAUNCH_SCRIPT \
     
# Group override parameters
override_params=(
  run_name=${run_name}
  data_local=${data_local}
  eval_loader.dataset.split=${eval_split_name}
  global_train_batch_size=${global_train_batch_size}
  device_train_microbatch_size=${device_train_microbatch_size}
  device_eval_batch_size=${device_eval_batch_size}
  max_seq_len=${max_seq_len}
  max_duration=${max_duration}
  eval_first=${eval_first}
  save_folder=${save_dir}
  loggers.wandb.init_kwargs.dir=${wandb_dir}
  eval_interval=${eval_interval}
  save_interval=${save_interval}
  optimizer.lr=${lr}
  model.l0_module=null
  model.path=${path}
  callbacks.data_loading.dynamic=${dynamic}
  callbacks.data_loading.set_names=${set_names}
  callbacks.data_loading.proportion=${proportion}
  callbacks.data_loading.update_type=${update_type}
  callbacks.data_loading.target_loss=${target_loss}
  train_loader.num_workers=0 # automatically use resources available in the current environment
  train_loader.prefetch_factor=null
  train_loader.persistent_workers=false
  autoresume=${CONFIG_AUTORECOVER}
)

if [[ "${CONFIG_AUTORECOVER}" = true ]]; then
  override_params+=(load_path=${LOAD_PATH})
fi

# Execute training
composer_cmd=(composer "$TRAIN_SCRIPT" $DEBUG_FLAG "$config_file")
composer_cmd+=("${override_params[@]}")

"${composer_cmd[@]}"
# checking eval_first