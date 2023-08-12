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
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--seed", type=int, help="Seed for random number generator", default=0)
    parser.add_argument("--prompt", type=str, help="Prompt for image generation")
    parser.add_argument("--mode", type=str, help="Mode for image generation", choices=["train","test"], default="train")
    parser.add_argument("--num_train_images", type=int, help="Number of images to generate for training", default=1000)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    checkpoints_dir = "/scratch/km3888/model_editing_attack_ckpt/textual_inversion/esd"
    
    gen = torch.Generator(device)
    gen.manual_seed(0)
    
    for model_ckpt in os.listdir(checkpoints_dir):
        prompt = ["A painting in the style of <art-style>"]*50
        print(model_ckpt)
        pipe = StableDiffusionPipeline.from_pretrained(os.path.join(checkpoints_dir, model_ckpt), safety_checker=None).to(device)
        
        os.makedirs(os.path.join("/scratch/mp5847/generated_images", model_ckpt), exist_ok=True)
        out = pipe(prompt=prompt, generator=gen, guidance_scale=7.5)
        
        for i, img in enumerate(out.images):
            img.save(os.path.join("/scratch/mp5847/generated_images", model_ckpt, f"{i}.png"))
