import torch
from PIL import Image
import open_clip
import os

if __name__ == "__main__":
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    mode = "text" #image or text

    human_eval_path = "/scratch/mp5847/human_eval_text"
    artists_dict = {
        "ajin_demi_human": "Ajin: Demi-Human",
        "kelly_mckernan": "Kelly McKernan",
        "kilian_eng": "Kilian Eng",
        "thomas_kinkade": "Thomas Kinkade",
        "tyler_edlin": "Tyler Edlin",
        "van_gogh": "Van Gogh",
    }
    
    #compute embeddings for real images
    real_images_embeddings = {}
    if mode == "image":
        for artist in artists_dict.keys():
            embeddings = []
            for img in os.listdir(os.path.join(human_eval_path, "real", "art", artist)):
                if img.endswith(".png"):
                    image = preprocess(Image.open(os.path.join(human_eval_path, "real", "art", artist, img))).unsqueeze(0)
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        image_features = model.encode_image(image)
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        embeddings.append(image_features)
            real_images_embeddings[artist] = torch.cat(embeddings, dim=0).mean(dim=0)
    elif mode == "text":
        for artist in artists_dict.keys():
            embeddings = []
            text = artists_dict[artist]
            with torch.no_grad(), torch.cuda.amp.autocast():
                text_features = model.encode_text(tokenizer(text))
                text_features /= text_features.norm(dim=-1, keepdim=True)
                embeddings.append(text_features)
            real_images_embeddings[artist] = torch.cat(embeddings, dim=0)
        
    print("Real images embeddings computed")
    
    #now let's compute embeddings for generated images
    for method in ["ac", "esd", "np", "sa", "sld_max", "sld_medium", "sld_strong", "sld_weak", "uce", "fmn"]:
        for artist in artists_dict.keys():
            print("Computing embeddings for {} {} images".format(method, artist))
            print("Number of images: {}".format(len(os.listdir(os.path.join(human_eval_path, method, "art", artist, "erased")))))
            print("Number of images: {}".format(len(os.listdir(os.path.join(human_eval_path, method, "art", artist, "ti")))))

            #check image for ti
            sim_history_ti = {}
            for img_path in os.listdir(os.path.join(human_eval_path, method, "art", artist, "ti")):
                if img_path.endswith(".png"):
                    img = preprocess(Image.open(os.path.join(human_eval_path, method, "art", artist, "ti", img_path))).unsqueeze(0)
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        img_features = model.encode_image(img)
                        img_features /= img_features.norm(dim=-1, keepdim=True)

                        #compute similarity of img_features and real_images_embeddings[artist]
                        sim = (img_features * real_images_embeddings[artist]).sum().item()
                        
                        
                sim_history_ti[img_path] = sim
            
            #keep top 10 and remove the rest
            sorted_sim = sorted(sim_history_ti.items(), key=lambda x: x[1], reverse=True)
            for i in range(10, len(sorted_sim)):
                os.remove(os.path.join(human_eval_path, method, "art", artist, "ti", sorted_sim[i][0]))

            #check image for erased
            sim_history_erased = {}
            for img_path in os.listdir(os.path.join(human_eval_path, method, "art", artist, "erased")):
                if img_path.endswith(".png"):
                    img = preprocess(Image.open(os.path.join(human_eval_path, method, "art", artist, "erased", img_path))).unsqueeze(0)
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        img_features = model.encode_image(img)
                        img_features /= img_features.norm(dim=-1, keepdim=True)

                        #compute similarity of img_features and real_images_embeddings[artist]
                        sim = (img_features * real_images_embeddings[artist]).sum().item()
                        
                sim_history_erased[img_path] = sim
            
            #keep bottom 10 and remove the rest
            sorted_sim = sorted(sim_history_erased.items(), key=lambda x: x[1], reverse=False)
            for i in range(10, len(sorted_sim)):
                os.remove(os.path.join(human_eval_path, method, "art", artist, "erased", sorted_sim[i][0]))
            
                
    #filter sd
    for artist in artists_dict.keys():
        sim_history_sd = {}
        for img_path in os.listdir(os.path.join(human_eval_path, "sd", "art", artist)):
            if img_path.endswith(".png"):
                img = preprocess(Image.open(os.path.join(human_eval_path, "sd", "art", artist, img_path))).unsqueeze(0)
                with torch.no_grad(), torch.cuda.amp.autocast():
                    img_features = model.encode_image(img)
                    img_features /= img_features.norm(dim=-1, keepdim=True)

                    #compute similarity of img_features and real_images_embeddings[artist]
                    sim = (img_features * real_images_embeddings[artist]).sum().item()
                    
            sim_history_sd[img_path] = sim
        
        #keep bottom 10 and remove the rest
        sorted_sim = sorted(sim_history_sd.items(), key=lambda x: x[1], reverse=False)
        for i in range(10, len(sorted_sim)):
            os.remove(os.path.join(human_eval_path, "sd", "art", artist, sorted_sim[i][0]))