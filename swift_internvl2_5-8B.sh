
export NPROC_PER_NODE=4 \
export CUDA_VISIBLE_DEVICES=0,1,2,3 \

swift sft \
    --model /data2/mjx/fine_tune/InternVL2_5-8B/ \
    --train_type lora \
    --freeze_vit True \
    --truncation_strategy right \
    --dataset '/data2/mjx/fine_tune/swift_train.json' \
    --split_dataset_ratio 0 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-5 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 2 \
    --save_steps 800 \
    --save_total_limit 1 \
    --logging_steps 20 \
    --max_length 4096 \
    --output_dir output \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --deepspeed zero3
