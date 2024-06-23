# a walk through latent space check out how to: https://keras.io/examples/generative/random_walks_with_stable_diffusion/
import random
import numpy as np
from argparse import ArgumentParser
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from img2vec_pytorch import Img2Vec
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




##trying_trained from ulrikkeeyer i google colab

#vae for ldm (latent diffusion model)
#conditions: 
no_train = False
batch_size = 16
n_epochs = 50
lr = 0.0001
store_path = "ddpm_fer2013_2.pt"
c, h, w = 1, 64, 64
device = "cuda" if torch.cuda.is_available() else "cpu"

start_ch = 1
start_h = 64
chs = [start_ch, start_ch*10,start_ch*10*2, start_ch*10*2*2, start_ch*10*2*2*2]
h_w = [start_h / 2, start_h / 2 / 2, start_h / 2 / 2 / 2]

images  = torch.load('efficientnet_b7_img_emo_emb.pt') #from  image_embeddings_img2vec
imgs = [] #images
classes = [] #classes
embeddings = []
for im in images:
  imgs.append(im[0])
  classes.append(im[1])
  embeddings.append(im[2])

print("shape of imgs", imgs[0].shape, "shape of class", classes[0].shape)

  
images0 = torch.cat(imgs, dim=0).view(-1, 3, 64, 64)
clss = torch.cat(classes, dim=0).view(-1, 1)
embeddings = torch.cat(embeddings, dim=0).view(-1, 512)
print(embeddings.shape, clss.shape)


class CustomDataset2(Dataset):
    def __init__(self):
        self.data = images0
        self.classes = clss
        self.embs = embeddings
        self.img_dim = (64, 64) 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        cls = self.classes[idx]
        to_pil = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()

        pil_image = to_pil(img) 
        transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize to 64x64
        transforms.Grayscale(),  # Convert to grayscale if not already
        transforms.ToTensor()  # Convert to tensor
        ])
   #     img_pil = transforms.functional.rgb_to_grayscale(pil_image)
   #     img_tensor = to_tensor(img_pil)
       # img_tensor = F.interpolate(img.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False)
       # img_tensor = img_tensor.squeeze(0)
        imgg = transform(pil_image)

       # try:
      #    img = Image.open(img_path)
     #     img_tensor = transform(img)
         # img_tensor = img_tensor.expand(3, -1, -1)
   #     except Exception as e:
   #       print("Error occurred:", e)
   #     class_id = self.class_map[class_name]
   #     img_tensor = img_tensor#.unsqueeze(0)
   #     class_id = torch.tensor([class_id])
        emb = self.embs[idx]
        return [imgg, cls], emb

data2 = CustomDataset2()
loader = DataLoader(data2, batch_size, shuffle=True)


for batch in loader:
    print("shape of images:", batch[0][0].shape, "shape of classes:", batch[0][1].shape, "shape of img embeds: ", batch[1].shape)
    break



################################################################### 
######### model ############################


class EMA:
    #hentet fra den youtubefillen. https://www.youtube.com/watch?v=TBCRlnwJtZU
    def __init__(self):
        super().__init__()
        self.beta = 0.995
        self.step = 0

    def reset_parameters(self, ema_model, model):
      ema_model.load_state_dict(model.state_dict())
    
    
    def update_model_average(self, ma_model, current_model):
      for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
          old_weight, up_weight = ma_params.data, current_params.data
          ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


    def step_ema(self, ema_model, model, step_start_ema=2000):
      if self.step < step_start_ema:
         self.reset_parameters(ema_model, model)
         self.step += 1
         return
      self.update_model_average(ema_model, model)
      self.step +=1



##################################################################################################################################
#########################################################    CROSS - ATTENTION    ################################################
##################################################################################################################################



class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = dim
        dim_out = dim_out
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)
# found it from here: https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

class CrossAttention(nn.Module):
  def __init__(self, embed_dim, hidden_dim, context_dim=None, num_heads=1,): #embed_dim: dim til bilde(1 eller 3 osv) , hiddem_dim:dim til bilde, context_dim:
    super(CrossAttention, self).__init__()
    self.hidden_dim = hidden_dim
    self.context_dim = context_dim
    self.embed_dim = embed_dim
    self.num_heads = self.embed_dim // 5 if embed_dim != 1 else 1 #3
    print(self.num_heads)
    self.multihead_attn = MultiHeadAttention(self.embed_dim,self.num_heads)

    self.query = nn.Linear(hidden_dim, embed_dim, bias=False)
    if context_dim is None:
      self.self_attn = True
      self.key = nn.Linear(hidden_dim, embed_dim, bias=False)     ###########
      self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)  ############
    else:
      self.self_attn = False
      self.key = nn.Linear(context_dim, embed_dim, bias=False)   #############
      self.value = nn.Linear(context_dim, hidden_dim, bias=False) ############


  def forward(self, tokens, context=None): #context:
    # tokens: with shape [batch, sequence_len, hidden_dim]
    # context: with shape [batch, contex_seq_len, context_dim]
    # x = (16,64*64,1)
    # label = (16,100)
   # print("x shape in attn is", tokens.shape)
    if self.self_attn:
        Q = self.query(tokens)
        K = self.key(tokens)
        V = self.value(tokens)
        return self.multihead_attn(Q,K,V)
    else:
        # implement Q, K, V for the Cross attention
        Q = self.query(tokens)
        K = self.key(context)
        V = self.value(context)
        return self.multihead_attn(Q,K,V)
   # print(Q.shape, K.shape, V.shape) #torch.Size([16, 4096, 1]) torch.Size([16, 1, 1]) torch.Size([16, 1, 1])
    ####### YOUR CODE HERE (2 lines)
    #print(Q.shape, K.shape)
   # scoremats = einsum('b i d, b j d -> b i j', Q, K)        # inner product of Q and K, a tensor
  #  attnmats = scoremats.softmax(dim=-1)        # softmax of scoremats
    #print(scoremats.shape, attnmats.shape, )
   # ctx_vecs = torch.einsum("BTS,BSH->BTH", attnmats, V)  # weighted average value vectors by attnmats
  #  print("regular crossattn shape output: ", ctx_vecs.shape)
   # return  ctx_vecs #out: (16, 64*64, 1)


## x = (28,1,28,28) , context = (28,1,512) ----> output: (28,1,28,28) --> basically samme som x input
# x = (16,64*64,1) label = (16,100)
#hidden dim: dimensjonen til bilde, som i dette tilfelle er 1 : (16,1,64,64) channel til bildet her altså
class TransformerBlock(nn.Module):
  def __init__(self, hidden_dim, context_dim=None):
    super(TransformerBlock, self).__init__()
    self.attn_self = CrossAttention(hidden_dim, hidden_dim, )
    if context_dim != None:
      self.attn_cross = CrossAttention(hidden_dim, hidden_dim, context_dim)
    else:
      self.attn_cross = CrossAttention(hidden_dim, hidden_dim,)


    self.norm1 = nn.LayerNorm(hidden_dim) #normalize på c,h,w per batch
    self.norm2 = nn.LayerNorm(hidden_dim)
    self.norm3 = nn.LayerNorm(hidden_dim)
    self.ffn  = FeedForward(hidden_dim, hidden_dim)


  def forward(self, x, context=None):
    x = self.attn_self(self.norm1(x)) + x
    if context != None:
    #  print("label is not none! shape is: ", context.shape)
      x = self.attn_cross(self.norm2(x), context=context) + x #size: [16, 256, 64] (batch, context_dim, h*w)
    x = self.ffn(self.norm3(x)) + x
    return x


class SpatialTransformer(nn.Module):
  def __init__(self, hidden_dim, context_dim): #hiddem_dim(channel for bildene, altså 1 eller 3 også oppover)
    super(SpatialTransformer, self).__init__()
    self.transformer = TransformerBlock(hidden_dim, context_dim)

  def forward(self, x, context=None): #xshape : 16,1,64,64 og contextshape: 16,100
    b, c, h, w = x.shape #(16,1,64,64)
    x_in = x
    # Combine the spatial dimensions and move the channel dimen to the end
    x = rearrange(x, "b c h w->b (h w) c") #(16, 64x64, 1)
    # Apply the sequence transformer
    x = self.transformer(x, context)
    # Reverse the process
    x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
    # Residue
    return x + x_in
  




##########################################################################################################################################################
####################################################  time-embeddings #####################################################################
#############################################################################################################################################################


def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:,::2] = torch.sin(t * wk[:,::2])
    embedding[:,1::2] = torch.cos(t * wk[:,::2])

    return embedding


#bare fra dimensjonene til T og til 1 slik at kan addes med x
def _make_te(dim_in, dim_out):
  return nn.Sequential(
    nn.Linear(dim_in, dim_out),
    nn.SiLU(),
    nn.Linear(dim_out, dim_out)
  )


##########################################################################################################################################################
####################################################  COND-DDPM #####################################################################
#############################################################################################################################################################


class CondDDPM(nn.Module):
    def __init__(self, network, n_steps=1000, min_beta=10 ** -4, max_beta=0.02, device=None, image_chw=(1, 64, 64)):
        super(CondDDPM, self).__init__()
        self.n_steps = n_steps
        self.device = device
        self.image_chw = image_chw
        self.cond_unet = network.to(device)
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        self.alphas = (1 - self.betas).to(device)
        self.alpha_bars = []
        for i, a in enumerate(self.alphas):
          if i == 0:
            self.alpha_bars.append(a)
          else:
            self.alpha_bars.append(torch.prod(self.alphas[:i + 1]))
        self.alpha_bars = torch.tensor(self.alpha_bars).to(device)


    def forward(self, x0, t, eta=None):
      n, c, h, w = x0.shape
      if eta is None:
        eta = torch.randn(n, c, h, w).to(self.device)
      part_one = self.alpha_bars[t].sqrt().reshape(n, 1, 1, 1) * x0
      part_two = (1 - self.alpha_bars[t].reshape(n, 1, 1, 1)).sqrt() * eta
      return part_one + part_two

    def backward(self, x, t, cond=None):
      return self.cond_unet(x,t, cond)
    


##########################################################################################################################################################
####################################################  ResNet Block #####################################################################
#############################################################################################################################################################

class ResNetBlock(nn.Module):
  def __init__(self, shape, in_c, out_c, block1=False):
        super(ResNetBlock, self).__init__()
        self.block1 = block1
        self.ln = nn.LayerNorm(shape)
        self.silu = nn.SiLU()
        self.relu = nn.ReLU()


        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False) #default stride=1
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False)

  def forward(self, x):
      out = self.ln(x)
      out = self.conv1(out)
      out = self.silu(out)
      out = self.conv2(out)
      #out = self.silu(out)
      if self.block1==True:
        return self.silu(out + x) #(x*(1/math.sqrt(2))))
      else:
        return self.silu(out)
      

class OneBlock(nn.Module):
    def __init__(self, context_dim):
        super(OneBlock, self).__init__()
        self.context_dim = context_dim
        self.block1 = ResNetBlock((chs[0],64,64), chs[0], chs[0], block1=True)
        self.attn = SpatialTransformer(chs[0], self.context_dim) #1,100
        self.block2 = ResNetBlock((chs[0],64,64), chs[0], chs[1])

    def forward(self, x, label):
        x = self.block1(x)
        x = self.attn(x, label)
        x = self.block2(x) #out: 10,28,28
        return x
##########################
##Out: [128, 10, 28, 28]##
##########################

class TwoBlock(nn.Module):
    def __init__(self, context_dim):
        super(TwoBlock, self).__init__()
        self.context_dim = context_dim
        self.block1 = ResNetBlock((chs[1],32,32), chs[1], chs[1], block1=True)
        self.attn = SpatialTransformer(chs[1], self.context_dim) #10,100
        self.block2 = ResNetBlock((chs[1],32,32), chs[1], chs[2])

    def forward(self, x, label):
        x = self.block1(x)
        x = self.attn(x, label)
        x = self.block2(x) #out: 20,14,14
        return x


##########################
##Out: [128, 20, 14, 14]##
##########################

class ThreeBlock(nn.Module):
    def __init__(self, context_dim):
        super(ThreeBlock, self).__init__()
        self.context_dim = context_dim
        self.block1 = ResNetBlock((chs[2],16,16), chs[2], chs[2], block1=True)
        self.attn = SpatialTransformer(chs[2], self.context_dim) #20,100
        self.block2 = ResNetBlock((chs[2],16,16), chs[2], chs[3])

    def forward(self, x, label):
        x = self.block1(x)
        x = self.attn(x, label)
        x = self.block2(x) #out: 40,7,7
        return x
    

###Bottleneck

class BottleNeck(nn.Module):
    def __init__(self, context_dim):
        super(BottleNeck, self).__init__()
        self.context_dim = context_dim
        self.block1 = ResNetBlock((chs[3],8,8), chs[3], chs[2])
        self.attn = SpatialTransformer(chs[2], self.context_dim) # 20,100
        self.block2 = ResNetBlock((chs[2],8,8), chs[2], chs[3])

    def forward(self, x, label):
        x = self.block1(x)
        x = self.attn(x, label)
        x = self.block2(x) #out: 40,3,3
        return x
    

class OneUp(nn.Module):
    def __init__(self, context_dim):
        super(OneUp, self).__init__()
        self.context_dim = context_dim
        self.block1 = ResNetBlock((chs[4],16,16), chs[4], chs[3])
        self.attn = SpatialTransformer(chs[3], self.context_dim) #40,100
        self.block2 = ResNetBlock((chs[3],16,16), chs[3], chs[2])

    def forward(self, x, label):
        x = self.block1(x)
        x = self.attn(x, label)
        x = self.block2(x) #out: 20,7,7
        return x


class TwoUp(nn.Module):
    def __init__(self, context_dim):
        super(TwoUp, self).__init__()
        self.context_dim = context_dim
        self.block1 = ResNetBlock((chs[3],32,32), chs[3], chs[2])
        self.attn = SpatialTransformer(chs[2], self.context_dim)
        self.block2 = ResNetBlock((chs[2],32,32), chs[2], chs[1])

    def forward(self, x, label):
        x = self.block1(x)
        x = self.attn(x, label)
        x = self.block2(x) #out: 10,14,14
        return x


class ThreeUp(nn.Module):
    def __init__(self, context_dim):
        super(ThreeUp, self).__init__()
        self.context_dim = context_dim
        self.block1 = ResNetBlock((chs[2],64,64), chs[2], chs[1])
        self.attn = SpatialTransformer(chs[1], self.context_dim)
        self.block2 = ResNetBlock((chs[1],64,64), chs[1], chs[1])

    def forward(self, x, label):
        x = self.block1(x)
        x = self.attn(x, label)
        x = self.block2(x) #out: 10,28,28
        return x

class ConditionalUNet(nn.Module):
  def __init__(self, n_steps=1000, time_emb_dim=512, context_dim=None): #switch time emb fra 512 til 100
        super(ConditionalUNet, self).__init__()
        # Sinusoidal embedding
        self.p_uncond = 0.2
        self.time_embed = nn.Embedding(n_steps, time_emb_dim) #embedding layer 1000 nedover, 100 bortover
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)
        self.label_enc = nn.Embedding(7, 512)
        self.context_dim = context_dim
        self.time_emb_dim = 512#100 + 512

        self.te1 = _make_te(self.time_emb_dim, chs[0])
        self.b1 = OneBlock(self.context_dim)
        self.down1 = nn.Conv2d(chs[1], chs[1], 4, 2, 1)

        self.te2 = _make_te(self.time_emb_dim, chs[1])
        self.b2 = TwoBlock(self.context_dim)
        self.down2 = nn.Conv2d(chs[2], chs[2], 4, 2, 1)

        self.te3 = _make_te(self.time_emb_dim, chs[2])
        self.b3 = ThreeBlock(self.context_dim)
        self.down3 = nn.Sequential(
        nn.Conv2d(chs[3], chs[3], 3, stride=2, padding=1),  # Adjust kernel size to 3x3 with stride 2
        nn.SiLU(),
        nn.Conv2d(chs[3], chs[3], 3, stride=1, padding=1)  # Adjust kernel size to 3x3 with stride 1
        )

        #BOTTLENECK
        self.te_mid = _make_te(self.time_emb_dim, chs[3])
        self.b_mid = BottleNeck(self.context_dim) #out: 40,3,3
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(chs[3], chs[3], 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(chs[3], chs[3], 3, 1, 1)
        ) #out: 40,7,7


        #decoders: 80 fordi plusser med de andre
        self.te4 = _make_te(self.time_emb_dim, chs[4])
        self.b4 = OneUp(self.context_dim) #out : 20, 7, 7
        self.up2 = nn.ConvTranspose2d(chs[2], chs[2], 4, 2, 1)

        self.te5 = _make_te(self.time_emb_dim, chs[3])
        self.b5 = TwoUp(self.context_dim) #out: 10,14,14
        self.up3 = nn.ConvTranspose2d(chs[1], chs[1], 4, 2, 1)

        self.te_out = _make_te(self.time_emb_dim, chs[2])
        self.b_out = ThreeUp(self.context_dim) #out: 10,28,28

        self.conv_out = nn.Conv2d(chs[1], chs[0], 3, 1, 1) #out: 1,28,28


#under sampling: kommer til å sette label til None
  def forward(self, x, t,  label=None): #clss=None
      t = self.time_embed(t) #for hver datapunkt i 16 sample, får sin t emb ift hvilket t den fikk, shape (16,100)
      
      ###bare under trening og en del av samplingen er label ikke lik None
      if label != None:
        print("SHAPE OF CLASS:", "SHAPE OF TIME:", t.shape, "SHAPE OF EMBEDDINGS:", label.shape)
        n, c, h, w = x.shape
       # label = self.label_enc(label) # får nå også shape (16,100)
        print(label.shape) 
        if label.shape == (n, 1):
           label = self.label_enc(label) #emo-embeddings shape: 16,1,512
        p_uncond = self.p_uncond
        value = int(x.shape[0] * 0.2) #how many we put to zero
        lst = [i for i in range(x.shape[0])] #0 til batch_size
        zero_indices = random.sample(lst, value) #velger 0.2*16=3 verdier mellom 0 til 15, altså hvilke indeksen som skal bli satt til 0
        for zero in zero_indices:
          label[zero] = 0 #batch.size*0.2 of batch will be set to zero
      ##########

        n = len(x)
        if label.shape == (16, 512): #skjekke for emo-embeddings vs image-embeddings
          label = label.unsqueeze(1) #image-emb shape (16,512) derfor trenger den å gå inn i denne
        t+=label
        #t = torch.cat((t, label), dim=2) #concat: [60, 1, 612] , add: [60, 1, 512]
       
      #t+=cls
      n = len(x)
      print(t.shape, label.shape, x.shape)
      out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1), label) 
      out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1), label)  # (N, 20, 14, 14)
      out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1), label)  # (N, 40, 7, 7)

      out_mid = self.b_mid(self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1), label)  # (N, 40, 3, 3)

      out4 = torch.cat((out3, self.up1(out_mid)), dim=1)  # (N, 80, 7, 7)
      out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1, 1), label)  # (N, 20, 7, 7)
      out5 = torch.cat((out2, self.up2(out4)), dim=1)  # (N, 40, 14, 14)
      out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1), label)  # (N, 10, 14, 14)
      out = torch.cat((out1, self.up3(out5)), dim=1)  # (N, 20, 28, 28)
      out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1), label)  # (N, 1, 28, 28)
      out = self.conv_out(out)
      
      return out



############################################################################################################################
###################################################  INFERENCE   ############################################################
############################################################################################################################


# Defining model
n_steps, min_beta, max_beta = 1000, 10 ** -4, 0.02  # Originally used by the authors
ddpm = CondDDPM(ConditionalUNet(context_dim=512), device=device)
ema = EMA()
ema_model = copy.deepcopy(ddpm).eval().requires_grad_(False).to(device)

def training_loop(ddpm, loader, n_epochs, device, display=False, store_path="ddpm_fer2013_2.pt"):
    mse = nn.MSELoss()
    best_loss = float("inf")
    n_steps = ddpm.n_steps
    optim = torch.optim.Adam(ddpm.parameters(), lr=lr)

    for epoch in tqdm(range(n_epochs), desc=f"Training progress", colour="#00ff00"):
        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(loader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour="#005500")):
          if step != len(loader) - 1:
            # Loading data
            x0 = batch[0][0].to(device)
            print("shape of img is:", x0.shape)
            clss = batch[0][1].to(device)
            print("shape of clss is:", clss.shape)
            labels = batch[1].to(device) #shape: 
            print("shape of labels is:", labels.shape)
            n = len(x0)
            encoded = x0
          
            # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
            eta = torch.randn_like(encoded).to(device)
            t = torch.randint(0, n_steps, (n,)).to(device) #shape (16,) tall mellom 0 og 1000

            # Computing the noisy image based on x0 and the time-step (forward process)
            noisy_imgs = ddpm(encoded, t, eta).to(device)

            # Getting model estimation of noise based on the images and the time-step
            eta_theta = ddpm.backward(noisy_imgs, t.reshape(n, -1), labels)
           # print("eta shape:", eta.shape, "eta_theta shape:", eta_theta.shape)

            # Optimizing the MSE between the noise plugged and the predicted noise
            loss = mse(eta_theta, eta)
            optim.zero_grad()
            loss.backward()
            optim.step()
            ema.step_ema(ema_model, ddpm)

            epoch_loss += loss.item() * len(x0) / len(loader.dataset)
       

        # Display images generated at this epoch
       # if display:
      #      show_images(generate_new_images(ddpm, device=device), f"Images generated at epoch {epoch + 1}")

        log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}"

        # Storing the model
       # if best_loss > epoch_loss:
        best_loss = epoch_loss
        torch.save(ddpm.state_dict(), store_path)
        torch.save(ema_model.state_dict(), "ddpm_efficienet_ema.pt")
        log_string += " --> Best model ever (stored)"

        print(log_string)
        
   

model = CondDDPM(ConditionalUNet(context_dim=512), n_steps=1000, device=device)

training_loop(model, loader, n_epochs, device, display=False, store_path="ddpm_efficienet.pt")

