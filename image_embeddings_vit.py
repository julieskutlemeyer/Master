from PIL import Image
import torch
from torchvision import transforms
import os
from tqdm import tqdm
#import ssl
#ssl._create_default_https_context = ssl._create_unverified_context
from pytorch_pretrained_vit import ViT
model = ViT('B_16_imagenet1k', pretrained=True)
model.eval()
#herfra: https://github.com/lukemelas/PyTorch-Pretrained-ViT

from torch import nn

from torchvision.models.vision_transformer import vit_b_16
from torchvision.models import ViT_B_16_Weights

from PIL import Image as PIL_Image

vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

preprocessing = ViT_B_16_Weights.DEFAULT.transforms()

transform = transforms.Compose([
    transforms.Resize((384, 384)), 
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
])
folder = "fer2013/train"
emotions =  os.listdir(folder)
print(emotions)


"""
path_emo_emb = []
for emo in emotions:
    if emo != ".DS_Store":
        paths = os.listdir(folder + "/" + emo)
        for path in tqdm(paths):
            full_path = folder + "/" + emo + "/" + path
            img = Image.open(full_path)
            img = transform(img).unsqueeze(0)
            with torch.no_grad():
                out = model(img)
                #print(outputs.shape)
                path_emo_emb.append([full_path, emo, out])

torch.save(path_emo_emb, "vit_lst_path_emo_emb.pt")
"""

path_emo_emb = []
for emo in tqdm(emotions):
    path_emo_emb = []
    if emo != ".DS_Store":
        paths = os.listdir(folder + "/" + emo)
        for path in tqdm(paths):
            full_path = folder + "/" + emo + "/" + path
            img = Image.open(full_path)
            img = img.convert('RGB')
            img = preprocessing(img)
            img = img.unsqueeze(0)
            
            feats = vit._process_input(img)
            batch_class_token = vit.class_token.expand(img.shape[0], -1, -1)
            feats = torch.cat([batch_class_token, feats], dim=1)
            feats = vit.encoder(feats)
            feats = feats[:, 0]
            path_emo_emb.append([full_path, emo, feats])
    torch.save(path_emo_emb, "vit_lst_path_"+ emo + "_emb.pt")
    print(emo + "tensors are now saved")


torch.save(path_emo_emb, "vit_lst_path_emo_emb.pt")



#from here: https://stackoverflow.com/questions/75874965/how-do-i-extract-features-from-a-torchvision-visitiontransfomer-vit

