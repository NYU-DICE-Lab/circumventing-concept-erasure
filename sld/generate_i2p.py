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
    parser.add_argument("--safety_concept", type=str, help="Text for negative prompt", default="")
    parser.add_argument("--sld_config", type=str, help="SLD config", default="none", choices=["none", "weak", "medium", "strong", "max"])
    parser.add_argument("--special_token", type=str, help="Special token for image generation", default="")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    metadata = []

    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "train"), exist_ok=True)

    data = load_dataset('AIML-TUDA/i2p', split='train')

    print("Number of images: ", len(data))
        
    pipe = SLDPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", safety_checker=None,
    ).to(device)    

    if model_path != "CompVis/stable-diffusion-v1-4": 
        text_encoder = CLIPTextModel.from_pretrained(args.model_path, subfolder="text_encoder", revision=False).to("cuda")
        tokenizer = CLIPTokenizer.from_pretrained(args.model_path, subfolder="tokenizer")
        pipe.text_encoder = text_encoder
        pipe.tokenizer = tokenizer
    pipe.safety_concept = args.safety_concept

    count = 0
    for item in data:
        if args.special_token != "":
            prompt = [item["prompt"]]
        else:
            prompt = [args.special_token + " " + item["prompt"]
        seed = item["sd_seed"]

        gen = torch.Generator(device)
        gen.manual_seed(seed)
        
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

        for image in out.images:
            
            file_name = "img_{}.png".format(count)
            image.save(os.path.join(args.output_dir, "train", file_name))
            count += 1

            metadata.append({"file_name": file_name, "prompt": prompt})

    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)
