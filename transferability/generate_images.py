from datasets import load_dataset
from diffusers import StableDiffusionPipeline
import torch
import os
import json
from PIL import Image
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images from I2P dataset")

    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--seed", type=int, help="Seed for random number generator", default=0)
    parser.add_argument("--prompt", type=str, help="Prompt for image generation")
    parser.add_argument("--save_name", type=str, default="train")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    gen = torch.Generator(device)

    os.makedirs(args.output_dir, exist_ok=True)

    gen.manual_seed(args.seed)

    pipe_og = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None).to(device)
    pipe_erase = StableDiffusionPipeline.from_pretrained("/scratch/mp5847/concept_erasure_iclr_ti_checkpoints/uce/art/van_gogh", safety_checker=None).to(device)

    pipe_erase.unet = pipe_og.unet
    
    out = pipe_erase(prompt=[args.prompt]*10, generator=gen)

    for i, img in enumerate(out.images):
        img.save(os.path.join(args.output_dir, f"{args.save_name}_{i}.png"))
            
        
    