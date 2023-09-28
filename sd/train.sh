export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATA_DIR="/scratch/mp5847/diffusers_generated_datasets/gwen_spiderman_6_real/train"
export OUTPUT_DIR="/scratch/mp5847/model_editing_attack_ckpt/textual_inversion/sd/gwen_spiderman_6_real"

accelerate launch concept_inversion.py \
        --pretrained_model_name_or_path=$MODEL_NAME \
        --train_data_dir=$DATA_DIR \
        --learnable_property="person" \
        --placeholder_token="<person>" --initializer_token="person" \
        --resolution=512 \
        --train_batch_size=1 \
        --gradient_accumulation_steps=4 \
        --max_train_steps=200 \
        --learning_rate=5.0e-03 --scale_lr \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --save_as_full_pipeline \
        --checkpointing_steps=200 \
        --output_dir=$OUTPUT_DIR \
        --num_train_images=6 \
        --mixed_precision="fp16" \
        --enable_xformers_memory_efficient_attention
        