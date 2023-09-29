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
    parser.add_argument("--safety_concept", type=str, help="Text for negative prompt", default="")
    parser.add_argument("--sld_config", type=str, help="SLD config", default="none", choices=["none", "weak", "medium", "strong", "max"])

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if(args.mode == "test"):

        gen = torch.Generator(device)

        os.makedirs(args.output_dir, exist_ok=True)

        gen.manual_seed(args.seed)
        
        pipe = SLDPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", safety_checker=None,
        ).to(device)    

        if model_path != "CompVis/stable-diffusion-v1-4": 
            text_encoder = CLIPTextModel.from_pretrained(args.model_path, subfolder="text_encoder", revision=False).to("cuda")
            tokenizer = CLIPTokenizer.from_pretrained(args.model_path, subfolder="tokenizer")
            pipe.text_encoder = text_encoder
            pipe.tokenizer = tokenizer
        pipe.safety_concept = args.safety_concept

        if(args.sld_config == "none"):
            out = pipe(prompt=args.prompt, generator=gen, sld_guidance_scale=0)
        elif(args.sld_config == "weak"):
            out = pipe(prompt=args.prompt, generator=gen,
            sld_warmup_steps=15,
            sld_guidance_scale=200,
            sld_threshold=0.0,
            sld_momentum_scale=0.0)
        elif(args.sld_config == "medium"):
            out = pipe(prompt=args.prompt, generator=gen,
            sld_warmup_steps=10,
            sld_guidance_scale=1000,
            sld_threshold=0.01,
            sld_momentum_scale=0.3,
            sld_mom_beta=0.4)
        elif(args.sld_config == "strong"):
            out = pipe(prompt=args.prompt, generator=gen,
            sld_warmup_steps=7,
            sld_guidance_scale=2000,
            sld_threshold=0.025,
            sld_momentum_scale=0.5,
            sld_mom_beta=0.7)
        elif(args.sld_config == "max"):
            out = pipe(prompt=args.prompt, generator=gen,
            sld_warmup_steps=0,
            sld_guidance_scale=5000,
            sld_threshold=1.0,
            sld_momentum_scale=0.5,
            sld_mom_beta=0.7)

        image = out.images[0]

        #save image
        image.save(os.path.join(args.output_dir, "img.png"))

    elif(args.mode == "train"):

        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir), exist_ok=True)
                
        pipe = SLDPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", safety_checker=None,
        ).to(device)    

        if model_path != "CompVis/stable-diffusion-v1-4": 
            text_encoder = CLIPTextModel.from_pretrained(args.model_path, subfolder="text_encoder", revision=False).to("cuda")
            tokenizer = CLIPTokenizer.from_pretrained(args.model_path, subfolder="tokenizer")
            pipe.text_encoder = text_encoder
            pipe.tokenizer = tokenizer
        pipe.safety_concept = args.safety_concept

        for i in range(args.num_train_images):
            while True:
                if(args.sld_config == "none"):
                    out = pipe(prompt=args.prompt, generator=gen, sld_guidance_scale=0)
                elif(args.sld_config == "weak"):
                    out = pipe(prompt=args.prompt, generator=gen,
                    sld_warmup_steps=15,
                    sld_guidance_scale=200,
                    sld_threshold=0.0,
                    sld_momentum_scale=0.0)
                elif(args.sld_config == "medium"):
                    out = pipe(prompt=args.prompt, generator=gen,
                    sld_warmup_steps=10,
                    sld_guidance_scale=1000,
                    sld_threshold=0.01,
                    sld_momentum_scale=0.3,
                    sld_mom_beta=0.4)
                elif(args.sld_config == "strong"):
                    out = pipe(prompt=args.prompt, generator=gen,
                    sld_warmup_steps=7,
                    sld_guidance_scale=2000,
                    sld_threshold=0.025,
                    sld_momentum_scale=0.5,
                    sld_mom_beta=0.7)
                elif(args.sld_config == "max"):
                    out = pipe(prompt=args.prompt, generator=gen,
                    sld_warmup_steps=0,
                    sld_guidance_scale=5000,
                    sld_threshold=1.0,
                    sld_momentum_scale=0.5,
                    sld_mom_beta=0.7)

                image = output.images[0]
                nsfw = output.nsfw_content_detected
                #check if nsfw is a list
                if isinstance(nsfw, list):
                    nsfw = nsfw[0]
                
                if not nsfw:
                    break
            image.save(os.path.join(args.output_dir, f"{args.prompt}_{i}.png"))
            
        
    