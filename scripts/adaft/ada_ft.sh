export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=ens32
export NCCL_DEBUG=INFO


LLM_VERSION="Qwen/Qwen2-0.5B-Instruct" 
# for 7b model we recommend bs=1, accum=2, 16 nodes, 128 gpus, lr=1e-5, warmup=0.03
# for 72b model we recommend bs=1, accum=1, 32 nodes, 256 gpus, lr=1e-5, warmup=0.03
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################

# BASE_RUN_NAME="llavanext-si_ai2d_gpt4v"
BASE_RUN_NAME="llavanext-si_alldata"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

############### Finetune ################

RUN_NUM="0_4cuda_0.3p"
# RUN_NUM="0_debug"
# Stage 2
PROMPT_VERSION="qwen_1_5"
RUN_NAME="llava-onevision-${BASE_RUN_NAME}_${RUN_NUM}" 
PREV_STAGE_CHECKPOINT="lmms-lab/llava-onevision-qwen2-0.5b-si" 
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${RUN_NAME}"

# export NUM_GPUS=4
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=0


# export LD_LIBRARY_PATH=/opt/conda/envs/adallava/lib:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH="/opt/conda/envs/adallava/lib:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/usr/local/lib:/usr/lib"


> ./scripts/adaft/adaft_stderr.log
> ./scripts/adaft/adaft_stdout.log

export WANDB_PROJECT=ada_llava_0.5b


# ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" \
deepspeed \
    llava/train/ada_train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path scripts/adaft/ada_si.yaml \
    --image_folder /home/ubuntu/projects/vqaData/data/llava_onevision \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
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
    --output_dir ./checkpoints/adaft/$RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10 \
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
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 32 \
    --ada_scheduler True \
    2> >(tee -a "./scripts/adaft/adaft_stderr.log" >&2) | tee -a "./scripts/adaft/adaft_stdout.log"
exit 0;

    # --ada_scheduler True \