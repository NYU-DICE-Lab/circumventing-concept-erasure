python3 generate_images_edit.py \
    --model_finetuned "/scratch/mp5847/diffusers_ckpt/tyler_edlin_50_sd_v1.4_fp16" \
    --prompts "a paiting in the style of Tyler Edlin" \
    --num_images 10 \
    --output_dir "./generated_images_tv_tyler_edlin" \
    --tv_edit_alpha 0.8