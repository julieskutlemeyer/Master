# a walk through latent space check out how to: https://keras.io/examples/generative/random_walks_with_stable_diffusion/
import random
import numpy as np
from argparse import ArgumentParser
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda
from torch import einsum
from einops import rearrange, repeat
from tqdm import tqdm
from torch.utils.data import Dataset
import cv2
import glob
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Lambda
from torchvision import transforms
import math 
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
import copy


images  = torch.load('images_64_fer2.pt') #from  image_embeddings_img2vec
images0 = [] #images
images1 = [] #classes
for im in images:
  images0.append(im[0])
  images1.append(im[1])

print("shape of imgs", images0[0].shape, "shape of class", images1[0].shape)
torch.save(images0[0][0], "one_img_batch.pt")
print("tensor saved, shape:", images0[0][0].shape)
embeddings  = torch.load('embeddings_64_fer2.pt')

torch.save(images0[0][0], "one_img_SAMEfer2013.pt")
torch.save(images1[0][0], "one_cls_SAMEfer2013.pt")
torch.save(embeddings[0][0], "one_emb_SAMEfer2013.pt")

