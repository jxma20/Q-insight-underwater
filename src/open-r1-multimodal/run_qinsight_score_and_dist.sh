set -x

export DEBUG_MODE="false"
RUN_NAME="score-and-dist-v5"
export LOG_PATH="./debug_log_$RUN_NAME.txt"
# v3: MUIQDtrain_dist + UIQD_score_desc + 3Epoch                          0.7672 0.6761 0.7293 2.4565
# v4: UIQD_dist + UIQD_score_desc + 3Epoch                                0.7474 0.6536 0.6883 2.6522
# v5: v3 + reward += 0.3 if nlp_score["rougeL"].fmeasure > 0.25 else 0


# set dist args
 SINGLE=1

nproc_per_node=${ARNOLD_WORKER_GPU}


if [ ! -z "$SINGLE" ] && [ "$SINGLE" != "0" ]; then
  echo "[single node alone] SINGLE=$SINGLE"
  nnodes=1
  node_rank=0
  nproc_per_node=8
  master_addr=127.0.0.1
  master_port=12345
else
  MASTER_NODE_ID=0
  nnodes=${ARNOLD_WORKER_NUM}
  node_rank=${ARNOLD_ID}
  master_addr="METIS_WORKER_${MASTER_NODE_ID}_HOST"
  master_addr=${!master_addr}
  master_port="METIS_WORKER_${MASTER_NODE_ID}_PORT"
  master_port=${!master_port}
  ports=(`echo $master_port | tr ',' ' '`)
  master_port=${ports[0]}
fi

echo "[nproc_per_node: ${nproc_per_node}]"
echo "[nnodes: ${nnodes}]"
echo "[node_rank: ${node_rank}]"
echo "[master_addr: ${master_addr}]"
echo "[master_port: ${master_port}]"


# set up envs
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0

export COMPILE_GAN=0
export USE_TIMELINE_SDK=1
export CUDA_TIMER_STREAM_KAFKA_CLUSTER=bmq_data_va
export CUDA_TIMER_STREAM_KAFKA_TOPIC=megatron_cuda_timer_tracing_original_v2
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

export DS_SKIP_CUDA_CHCK=1

torchrun --nproc_per_node=${nproc_per_node} \
    --nnodes=${nnodes} \
    --node_rank=${node_rank} \
    --master_addr=${master_addr} \
    --master_port=${master_port} \
    src/open_r1/qinsight_multi_task.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir output/$RUN_NAME \
    --model_name_or_path /HOME/paratera_xy/pxy1092/HDD_POOL/Q-Insight/Qwen2.5-VL-7B \
    --dataset_name None \
    --dataset_score data_config/iqa_score.yaml \
    --dataset_dist data_config/iqa_dist.yaml \
    --image_root ./data \
    --max_prompt_length 512 \
    --num_generations 8 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 3 \
    --run_name $RUN_NAME \
    --save_steps 1500 \
    --save_only_model true \
    --score_reward_threshold 0.35 \
    --beta 0.001

