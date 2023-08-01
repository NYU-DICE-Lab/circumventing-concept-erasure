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
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    device = 'cuda' if args.cuda else 'cpu'

    gen = torch.Generator(device)

    os.makedirs(args.output_dir, exist_ok=True)

    gen.manual_seed(args.seed)

    pipe = StableDiffusionPipeline.from_pretrained(args.model_path, safety_checker=None, torch_dtype=torch.float16).to(device)
    
    out = pipe(prompt=[args.prompt], generator=gen)

    image = out.images[0]

    #save image
    image.save(os.path.join(args.output_dir, "img.png"))

    