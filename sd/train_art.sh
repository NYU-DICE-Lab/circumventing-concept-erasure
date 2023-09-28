# ajin_demi_human
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATA_DIR="/scratch/mp5847/diffusers_generated_datasets/ajin_demi_human_real_6/train"
export OUTPUT_DIR="/scratch/mp5847/model_editing_attack_ckpt/textual_inversion/sd/ajin_demi_human_real_6"

accelerate launch concept_inversion.py \
        --pretrained_model_name_or_path=$MODEL_NAME \
        --train_data_dir=$DATA_DIR \
        --learnable_property="style" \
        --placeholder_token="<art-style>" --initializer_token="art" \
        --resolution=512 \
        --train_batch_size=4 \
        --gradient_accumulation_steps=4 \
        --max_train_steps=1000 \
        --learning_rate=5.0e-03 --scale_lr \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --save_as_full_pipeline \
        --checkpointing_steps=1000 \
        --output_dir=$OUTPUT_DIR \
        --num_train_images=6 \
        --mixed_precision="fp16" \
        --enable_xformers_memory_efficient_attention

# kelly_mckernan
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATA_DIR="/scratch/mp5847/diffusers_generated_datasets/kelly_mckernan_real_6/train"
export OUTPUT_DIR="/scratch/mp5847/model_editing_attack_ckpt/textual_inversion/sd/kelly_mckernan_real_6_50step"

accelerate launch concept_inversion.py \
        --pretrained_model_name_or_path=$MODEL_NAME \
        --train_data_dir=$DATA_DIR \
        --learnable_property="style" \
        --placeholder_token="<art-style>" --initializer_token="art" \
        --resolution=512 \
        --train_batch_size=4 \
        --gradient_accumulation_steps=4 \
        --max_train_steps=1000 \
        --learning_rate=5.0e-03 --scale_lr \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --save_as_full_pipeline \
        --checkpointing_steps=1000 \
        --output_dir=$OUTPUT_DIR \
        --num_train_images=6 \
        --mixed_precision="fp16" \
        --enable_xformers_memory_efficient_attention

# van_gogh
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATA_DIR="/scratch/mp5847/diffusers_generated_datasets/van_gogh_real_6/train"
export OUTPUT_DIR="/scratch/mp5847/model_editing_attack_ckpt/textual_inversion/sd/van_gogh_real_6"

accelerate launch concept_inversion.py \
        --pretrained_model_name_or_path=$MODEL_NAME \
        --train_data_dir=$DATA_DIR \
        --learnable_property="style" \
        --placeholder_token="<art-style>" --initializer_token="art" \
        --resolution=512 \
        --train_batch_size=4 \
        --gradient_accumulation_steps=4 \
        --max_train_steps=1000 \
        --learning_rate=5.0e-03 --scale_lr \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --save_as_full_pipeline \
        --checkpointing_steps=1000 \
        --output_dir=$OUTPUT_DIR \
        --num_train_images=6 \
        --mixed_precision="fp16" \
        --enable_xformers_memory_efficient_attention

# tyler_edlin
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATA_DIR="/scratch/mp5847/diffusers_generated_datasets/tyler_edlin_real_6/train"
export OUTPUT_DIR="/scratch/mp5847/model_editing_attack_ckpt/textual_inversion/sd/tyler_edlin_real_6"

accelerate launch concept_inversion.py \
        --pretrained_model_name_or_path=$MODEL_NAME \
        --train_data_dir=$DATA_DIR \
        --learnable_property="style" \
        --placeholder_token="<art-style>" --initializer_token="art" \
        --resolution=512 \
        --train_batch_size=4 \
        --gradient_accumulation_steps=4 \
        --max_train_steps=1000 \
        --learning_rate=5.0e-03 --scale_lr \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --save_as_full_pipeline \
        --checkpointing_steps=1000 \
        --output_dir=$OUTPUT_DIR \
        --num_train_images=6 \
        --mixed_precision="fp16" \
        --enable_xformers_memory_efficient_attention

# thomas_kinkade
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATA_DIR="/scratch/mp5847/diffusers_generated_datasets/thomas_kinkade_real_6/train"
export OUTPUT_DIR="/scratch/mp5847/model_editing_attack_ckpt/textual_inversion/sd/thomas_kinkade_real_6"

accelerate launch concept_inversion.py \
        --pretrained_model_name_or_path=$MODEL_NAME \
        --train_data_dir=$DATA_DIR \
        --learnable_property="style" \
        --placeholder_token="<art-style>" --initializer_token="art" \
        --resolution=512 \
        --train_batch_size=4 \
        --gradient_accumulation_steps=4 \
        --max_train_steps=1000 \
        --learning_rate=5.0e-03 --scale_lr \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --save_as_full_pipeline \
        --checkpointing_steps=1000 \
        --output_dir=$OUTPUT_DIR \
        --num_train_images=6 \
        --mixed_precision="fp16" \
        --enable_xformers_memory_efficient_attention

# kilian_eng
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATA_DIR="/scratch/mp5847/diffusers_generated_datasets/kilian_eng_real_6/train"
export OUTPUT_DIR="/scratch/mp5847/model_editing_attack_ckpt/textual_inversion/sd/kilian_eng_real_6_50step"

accelerate launch concept_inversion.py \
        --pretrained_model_name_or_path=$MODEL_NAME \
        --train_data_dir=$DATA_DIR \
        --learnable_property="style" \
        --placeholder_token="<art-style>" --initializer_token="art" \
        --resolution=512 \
        --train_batch_size=4 \
        --gradient_accumulation_steps=4 \
        --max_train_steps=1000 \
        --learning_rate=5.0e-03 --scale_lr \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --save_as_full_pipeline \
        --checkpointing_steps=1000 \
        --output_dir=$OUTPUT_DIR \
        --num_train_images=6 \
        --mixed_precision="fp16" \
        --enable_xformers_memory_efficient_attention