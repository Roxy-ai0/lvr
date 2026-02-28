#!/bin/bash

# ==========================================
# 1. 基础环境与路径配置
# ==========================================
MODEL_NAME="/mnt/afs/smartbrain/fangwenhan/wzz/gzh/LatentReasoning_with_SAE/model/Qwen2.5-VL-7B-Instruct"
export WANDB_PROJECT="LVR-Qwen25-VL-7B-SFT-STAGE-1-450k"
export WANDB_API_KEY="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"

export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$PWD:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=3,4,6,7

DATA_PATH="/mnt/afs/smartbrain/fangwenhan/wzz/gzh/LatentReasoning_with_SAE/data/lvr_data/meta_data_lvr_sft_stage1.json"
OUTPUT_DIR="stage1_checkpoints/"

# ==========================================
# 2. 训练超参数
# ==========================================
DATA_PACKING=True
LST=4096
MAX_INSTANCE_PER_BATCH=2
MAX_PACKED_TOKENS=$((MAX_INSTANCE_PER_BATCH * LST))
RANDOM_SEED=42

BATCH_PER_DEVICE=1
NUM_DEVICES=4
GRAD_ACCUM_STEPS=16
LR=1e-5
LVR_HEAD=False
LVR_LOSS_FCT=mse
LAMBDA_LVR=0.1
MAX_TOKEN=5120
MIN_TOKEN=128

RUN_NAME="Stage1_${LVR_LOSS_FCT}LVRLossLambda${LAMBDA_LVR}-MaxVisToken${MAX_TOKEN}-MinVisToken${MIN_TOKEN}"
ONLINE=False

# ==========================================
# 3. 核心：分段训练与主动重启逻辑
# ==========================================
TOTAL_STEPS=2500      # 我们的最终总目标是 2500 步
RUN_STEPS=100         # 每次只跑 100 步就主动退出，释放内存！

while true; do
    # 扫描最新的 Checkpoint
    LATEST_CHECKPOINT=$(ls -d ${OUTPUT_DIR}checkpoint-* 2>/dev/null | sort -V | tail -n 1)

    if [ -n "$LATEST_CHECKPOINT" ]; then
        CHECKPOINT_BASENAME=$(basename $LATEST_CHECKPOINT)
        # 从 "checkpoint-100" 中提取出数字 "100"
        CURRENT_STEP=${CHECKPOINT_BASENAME#checkpoint-}
        RESUME_ARGS="--resume_from_checkpoint $LATEST_CHECKPOINT"
        echo "=================================================================="
        echo "🔍 发现历史检查点: $LATEST_CHECKPOINT (当前已跑 $CURRENT_STEP 步)"
    else
        CURRENT_STEP=0
        RESUME_ARGS=""
        echo "=================================================================="
        echo "🌱 未发现历史检查点，从第 0 步开始..."
    fi

    # 如果当前步数已经达到了 2500 步，说明大功告成，跳出死循环
    if [ "$CURRENT_STEP" -ge "$TOTAL_STEPS" ]; then
        echo "🎉 已经达到总目标 $TOTAL_STEPS 步，训练圆满完成！"
        break
    fi

    # 动态计算本次运行的终点 (例如当前 100，本次终点就是 200)
    CURRENT_TARGET=$((CURRENT_STEP + RUN_STEPS))
    
    # 防止最后一次超过 2500 步
    if [ "$CURRENT_TARGET" -gt "$TOTAL_STEPS" ]; then
        CURRENT_TARGET=$TOTAL_STEPS
    fi

    echo "🚀 本次计划跑到第 $CURRENT_TARGET 步，然后主动退出以清空内存..."
    echo "=================================================================="

    # 启动训练，将 max_steps 设为动态计算的 CURRENT_TARGET
    deepspeed --master_port 29501 src/train/train_lvr.py \
        --run_name "$RUN_NAME" \
        --coconut True \
        --loss_lvr_fct $LVR_LOSS_FCT \
        --deepspeed scripts/zero3_offload.json \
        --model_id $MODEL_NAME \
        --data_path "$DATA_PATH" \
        --remove_unused_columns False \
        --lvr_head $LVR_HEAD \
        --freeze_vision_tower True \
        --freeze_merger True \
        --freeze_llm False \
        --max_steps $CURRENT_TARGET \
        --learning_rate $LR \
        --loss_lvr_lambda $LAMBDA_LVR \
        --bf16 True \
        --fp16 False \
        --disable_flash_attn2 False \
        --online_checkpoint $ONLINE \
        --output_dir "$OUTPUT_DIR" \
        --num_train_epochs 1 \
        --per_device_train_batch_size $BATCH_PER_DEVICE \
        --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
        --image_min_pixels $((MIN_TOKEN * 28 * 28)) \
        --image_max_pixels $((MAX_TOKEN * 28 * 28)) \
        --weight_decay 0.1 \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 False \
        --gradient_checkpointing True \
        --report_to wandb \
        --lazy_preprocess True \
        --save_strategy "steps" \
        --save_steps $RUN_STEPS \
        --save_total_limit 10 \
        --dataloader_num_workers 4 \
        --enable_data_packing $DATA_PACKING \
        --max_packed_tokens $MAX_PACKED_TOKENS \
        --random_seed $RANDOM_SEED \
        --long_seq_threshold $LST \
        --max_instance_per_batch $MAX_INSTANCE_PER_BATCH \
        $RESUME_ARGS

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ 本轮 $RUN_STEPS 步已跑完，内存已释放。准备进行下一轮..."
        sleep 5
    else
        echo "⚠️ 训练出现异常 (退出码: $EXIT_CODE)，10秒后重启..."
        sleep 10
    fi
done