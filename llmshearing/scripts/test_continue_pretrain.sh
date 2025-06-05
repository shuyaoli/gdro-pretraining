# continue pre-train llama2 7b -> 3b or 1.3b

PROJ_DIR=$HOME/LLM-Shearing
DATA_DIR=${PROJ_DIR}/llm_data/LLM-Shearing/for_prune
OUTPUT_DIR=${PROJ_DIR}/models/pretrained
LAUNCH_SCRIPT=${PROJ_DIR}/llmshearing/scripts/launch.sh
TRAIN_SCRIPT=${PROJ_DIR}/llmshearing/train.py

eval_first=False
test=True
optimizer=Adam

model=1.3b # target model size
config_file=${PROJ_DIR}/llmshearing/configs/llama2/${model}_${optimizer}.yaml
prune_run_name=LLaMA-1-3-B-Pruned_${optimizer}
path=/home/shuyaoli/LLM-Shearing/models/LLaMA-1-3-B-Pruned/state_dict.pt # path to the pruned model

# data setup
data_local=${DATA_DIR}

# basic setup
max_seq_len=4096
device_train_microbatch_size=4
global_train_batch_size=256
device_eval_batch_size=8

# learning setup
lr=1e-4 # learning rate for the main parameters
max_duration=48000ba # 50B tokens
save_interval=3200ba # save every 3200ba
# t_warmup=1440ba # 3% learning rate warmup 

# dynamic loading setup
dynamic=True
set_names="[cc,github,book,stackexchange,wiki,arxiv,c4-rp]" # domain names
proportion="[0.2192,0.0002,0.0791,0.0064,0.0096,0.001,0.6845]" # final proportion of pruning
# doremi: update weights with exponential descent
# constant: keep the weights constant
update_type="pd-kl" 
target_loss="[1.9643,0.7459,2.1393,1.6117,1.7590,1.4449,2.1251]" # 1.3b predicted loss from scaling law

eval_split_name=eval_merge # eval on all domains
eval_interval=400ba # eval every 50 batches and update the loading proportion


# save directroy
run_name=${prune_run_name}_ft${max_duration}_${optimizer}_test
save_dir=${OUTPUT_DIR}/${run_name}
wandb_dir=${save_dir} # save locally

if [[ $test == True ]]; then t=00-1:00:00; else t=01-00:00:00; fi

# Run in bash, it will automatically use resources available in the current environment
# composer $TRAIN_SCRIPT \


# Run with slurm
# sbatch --job-name ${run_name} \
#     --nodes=8 \
#     --gpus-per-node=2 \
#     --mem=512gb \
#     --cpus-per-task=8 \
#     --time $t \
#     --dependency=singleton \
#     $LAUNCH_SCRIPT \
#     $config_file \
#     run_name=${run_name} \
#     data_local=${data_local} \
#     eval_loader.dataset.split=${eval_split_name} \
#     global_train_batch_size=${global_train_batch_size} \
#     device_train_microbatch_size=${device_train_microbatch_size} \
#     device_eval_batch_size=${device_eval_batch_size} \
#     max_seq_len=${max_seq_len} \
#     max_duration=${max_duration} \
#     eval_first=true \
#     scheduler.t_warmup=${t_warmup} \
#     save_folder=${save_dir} \
#     loggers.wandb.init_kwargs.dir=${wandb_dir} \
#     eval_interval=${eval_interval} \
#     save_interval=${save_interval} \
#     optimizer.lr=${lr} \
#     model.l0_module=null \
#     model.path=${path} \
#     callbacks.data_loading.dynamic=${dynamic} \
#     callbacks.data_loading.set_names=${set_names} \
#     callbacks.data_loading.proportion=${proportion} \
#     callbacks.data_loading.update_type=${update_type} \
#     callbacks.data_loading.target_loss=${target_loss} \
#     train_loader.num_workers=0 \
#     train_loader.prefetch_factor=null \
#     train_loader.persistent_workers=false \
#     autoresume=true

# checking eval_first

# Run in bash, it will automatically use resources available in the current environment
composer $TRAIN_SCRIPT \
    $config_file \
    run_name=${run_name} \
    data_local=${data_local} \
    eval_loader.dataset.split=${eval_split_name} \
    global_train_batch_size=${global_train_batch_size} \
    device_train_microbatch_size=${device_train_microbatch_size} \
    device_eval_batch_size=${device_eval_batch_size} \
    max_seq_len=${max_seq_len} \
    max_duration=${max_duration} \
    eval_first=${eval_first} \
    save_folder=${save_dir} \
    loggers.wandb.init_kwargs.dir=${wandb_dir} \
    eval_interval=${eval_interval} \
    save_interval=${save_interval} \
    optimizer.lr=${lr} \
    model.l0_module=null \
    model.path=${path} \
    callbacks.data_loading.dynamic=${dynamic} \
    callbacks.data_loading.set_names=${set_names} \
    callbacks.data_loading.proportion=${proportion} \
    callbacks.data_loading.update_type=${update_type} \
    callbacks.data_loading.target_loss=${target_loss} \
    train_loader.num_workers=0 \
    train_loader.prefetch_factor=null \
    train_loader.persistent_workers=false \
    autoresume=false \
    eval_loader.num_workers=4
    # scheduler.t_warmup=${t_warmup} \
# checking eval_first