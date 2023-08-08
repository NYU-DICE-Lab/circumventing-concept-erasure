export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATA_DIR=""
export OUTPUT_DIR="./"

accelerate launch concept_inversion.py \
        --pretrained_model_name_or_path=$MODEL_NAME \
        --train_data_dir=$DATA_DIR \
        --learnable_property="style" \
        --placeholder_token="<art-style>" --initializer_token="art" \
        --resolution=512 \
        --train_batch_size=1 \
        --gradient_accumulation_steps=4 \
        --max_train_steps=1000 \
        --learning_rate=5.0e-03 --scale_lr \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --save_as_full_pipeline \
        --checkpointing_steps=1000 \
        --output_dir=$OUTPUT_DIR \
        --num_train_images=25 \
        --safety_concept="nudity" \
        --timestep_range=10 \
        --timestep_count=1 \
        --i2p \
        --i2p_metadata_path="metadata.json" \
        --mixed_precision="fp16"
