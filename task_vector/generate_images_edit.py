from diffusers import StableDiffusionPipeline
import torch
import os
import json
from math import ceil, sqrt
from PIL import Image
from utils import save_image, concat_images_in_square_grid, TaskVector
import argparse

#add parser function
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_pretrained', type=str, default="CompVis/stable-diffusion-v1-4", help='pretrained model')
    parser.add_argument('--model_finetuned', type=str, default="", help='finetuned model')
    parser.add_argument('--prompts', type=str)
    parser.add_argument('--num_images', type=int, default=30, help='number of images')
    parser.add_argument('--output_dir', type=str, default="/scratch/mp5847/diffusers_ckpt/output", help='output directory')
    parser.add_argument('--tv_edit_alpha', type=float, default=0.5, help='amount of edit to task vector layer')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe_pretrained = StableDiffusionPipeline.from_pretrained(args.model_pretrained, torch_dtype=torch.float16, safety_checker=None)
    pipe_pretrained.to(device)

    os.makedirs(args.output_dir, exist_ok=True)

    print("Generating images ...")
    print("Edit prompt: ", args.prompts)

    pipe_finetuned = StableDiffusionPipeline.from_pretrained(args.model_finetuned, torch_dtype=torch.float16, safety_checker=None)
    pipe_finetuned.to("cuda")

    #edit process
    unet_pretrained = pipe_pretrained.unet
    unet_finetuned = pipe_finetuned.unet

    #save model unet
    torch.save(unet_pretrained, "unet_pretrained.pt")
    torch.save(unet_finetuned, "unet_finetuned.pt")

    task_vector = TaskVector(pretrained_checkpoint="unet_pretrained.pt", 
                            finetuned_checkpoint="unet_finetuned.pt")

    vector = task_vector.vector
        
    neg_task_vector = -task_vector
    unet_edited = neg_task_vector.apply_to("unet_pretrained.pt", scaling_coef=args.tv_edit_alpha)
    pipe_pretrained.unet = unet_edited

    os.remove("unet_pretrained.pt")
    os.remove("unet_finetuned.pt")

    #save model unet_edited
    torch.save(unet_edited.state_dict(), "unet_edited.pt")

    #save images
    gen = torch.Generator(device)
    gen.manual_seed(0)
    prompts = [args.prompts]*args.num_images
    out = pipe_pretrained(prompt=prompts, generator=gen)

    for i, img in enumerate(out.images):
        img.save(os.path.join(args.output_dir, f"{i}_{args.tv_edit_alpha}.png"))


    print("Done!")
    