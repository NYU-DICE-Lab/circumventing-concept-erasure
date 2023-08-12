export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="/scratch/mp5847/diffusers_generated_datasets/tyler_edlin_50_sd_v1.4"

accelerate launch train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name="$DATASET_NAME" \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=2 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=100 \
  --learning_rate=3e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --checkpointing_steps=100 \
  --output_dir="/scratch/mp5847/diffusers_ckpt/tyler_edlin_50_sd_v1.4_fp16" --mixed_precision "fp16"