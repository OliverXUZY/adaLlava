# ## debug on one GPU
# export OMP_NUM_THREADS=8
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO
# export PYTORCH_JIT=0
# export OMP_NUM_THREADS=8


LLM_VERSION="Qwen/Qwen2-7B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################
BASE_RUN_NAME="llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

############### Finetune ################
# Stage 2
PROMPT_VERSION="qwen_1_5"
RUN_NAME="debug_1gpu"
PREV_STAGE_CHECKPOINT="lmms-lab/llava-onevision-qwen2-0.5b-si" 
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${RUN_NAME}"

export LD_LIBRARY_PATH="/opt/conda/envs/adallava/lib:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/usr/local/lib:/usr/lib"


# Run on a single GPU
deepspeed --include localhost:0 llava/train/ada_train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path scripts/adaft/ada_MME.yaml \
    --eval_data_path /home/ubuntu/projects/vqaData/data/MME/json_qa/subset_qa_MME.json \
    --image_folder /home/ubuntu/projects/vqaData/data/MME/images \
    --mm_tunable_parts="lm_ada_scheduler" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir ./checkpoints/debug/$RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 1 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 32 \
    --log_level info \
    --ada_scheduler True \


        # --deepspeed scripts/zero3.json \
# --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \