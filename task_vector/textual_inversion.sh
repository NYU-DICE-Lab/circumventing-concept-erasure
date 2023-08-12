export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATA_DIR="/scratch/mp5847/diffusers_generated_datasets/tyler_edlin_50_sd_v1.4/train"
export OUTPUT_DIR="/scratch/mp5847/textual_inversion_ckpt_tyler_edlin"
export ESD_CKPT="/home/mp5847/src/circumventing-concept-erasure/task_vector/unet_edited.pt"

accelerate launch textual_inversion.py \
        --pretrained_model_name_or_path=$MODEL_NAME \
        --train_data_dir=$DATA_DIR \
        --learnable_property="style" \
        --placeholder_token="<art-style>" --initializer_token="art" \
        --resolution=512 \
        --train_batch_size=1 \
        --gradient_accumulation_steps=4 \
        --max_train_steps=1000 \
        --learning_rate=1.0e-03 --scale_lr \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --save_as_full_pipeline \
        --checkpointing_steps=1000 \
        --output_dir=$OUTPUT_DIR \
        --num_train_images=25 \
        --esd_checkpoint=$ESD_CKPT \
        --mixed_precision "fp16"