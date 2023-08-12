import torch
from PIL import Image
import open_clip
import os
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
import torch.nn as nn
import json

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_folders_path = "/scratch/mp5847/generated_images"
    ckpt_folders_path = "/scratch/km3888/model_editing_attack_ckpt/textual_inversion/esd"

    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k', device=device)

    metadata = []
    
    with torch.no_grad():

        for i, folder in enumerate(tqdm(os.listdir(img_folders_path))):
            img_folder_path = os.path.join(img_folders_path,folder)
            ckpt_folder_path = os.path.join(ckpt_folders_path,folder)
            tokenizer = CLIPTokenizer.from_pretrained(ckpt_folder_path, subfolder="tokenizer")
            text_encoder = CLIPTextModel.from_pretrained(ckpt_folder_path, subfolder="text_encoder").to(device)
            
            tokenizer_og = open_clip.get_tokenizer("ViT-L-14")

            img_score = []
            txt_score = []

            for img in os.listdir(img_folder_path):
                img_path = os.path.join(img_folder_path,img)
                image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
                if("van_gogh" in img_folder_path):
                    text = tokenizer(["<art-style>","Van Gogh style"],  max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
                    text_og = tokenizer_og(["<art-style>","Van Gogh style"]).to(device)
                elif("tyler_edlin" in img_folder_path):
                    text = tokenizer(["<art-style>","Tyler Edlin style"],  max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
                    text_og = tokenizer_og(["<art-style>","Tyler Edlin style"]).to(device)

                text_embeddings = text_encoder(text)[1]
                text_embeddings_og = model.encode_text(text_og)
                img_embeddings = model.encode_image(image)

                img_embeddings /= img_embeddings.norm(dim=-1, keepdim=True)
                text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
                text_embeddings_og /= text_embeddings_og.norm(dim=-1, keepdim=True)

                text_embeddings = text_embeddings.to(device)
                text_embeddings_og = text_embeddings_og.to(device)
                img_embeddings = img_embeddings.to(device)

                cosine_similarity1 = nn.CosineSimilarity(dim=1, eps=1e-6)(img_embeddings, text_embeddings_og[[1]])
                cosine_similarity2 = nn.CosineSimilarity(dim=1, eps=1e-6)(text_embeddings[[0]], text_embeddings[[1]])

                img_score.append(cosine_similarity1.item())
                txt_score.append(cosine_similarity2[0].item())
            
            item = {"folder":folder, "img_score":img_score, "text_score":sum(txt_score)/len(txt_score)}

            metadata.append(item)
    
    #save metadata as json
    with open("metadata.json", "w") as f:
        json.dump(metadata, f)
       