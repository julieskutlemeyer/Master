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
from img2vec_pytorch import Img2Vec
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
import torchvision.transforms.functional as TF

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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


###fant her
#### from here: https://medium.com/analytics-vidhya/read-fer2013-face-expression-recognition-dataset-using-pytorch-torchvision-9ff64f55018e
##and here: https://medium.com/analytics-vidhya/creating-a-custom-dataset-and-dataloader-in-pytorch-76f210a1df5d


class CustomDataset(Dataset):
    def __init__(self):
        self.imgs_path = "fer2013/test/"
        self.file_list = glob.glob(self.imgs_path + "*")
        #self.file_list = glob.glob(self.imgs_path + "*.jpg") + glob.glob(self.imgs_path + "*.jpeg")
        #print(self.file_list)
        self.data = []
        for class_path in self.file_list:
            class_name = class_path.split("/")[-1]
            for img_path in glob.glob(class_path + "/*.jpg"):
                self.data.append([img_path, class_name])
       # print(len(self.data), self.data)
        self.class_map = {'angry' : 0, 'disgust' : 1, 'fear' : 2, 'happy' : 3, 'sad' : 4, 'surprise' : 5, 'neutral' : 6}
        self.img_dim = (64, 64) #224
 #       for d in self.dat:
  #        img = cv2.imread(d[0])
  #        if img is not None:
  #          self.data.append(d)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize to 64x64
#        transforms.Grayscale(),  # Convert to grayscale if not already
        transforms.ToTensor()  # Convert to tensor
        ])
        try:
          img = Image.open(img_path)
          img_tensor = transform(img)
          img_tensor = img_tensor.expand(3, -1, -1)
        except Exception as e:
          print("Error occurred:", e)
        class_id = self.class_map[class_name]
        img_tensor = img_tensor#.unsqueeze(0)
        class_id = torch.tensor([class_id])
        return img_tensor, [class_id, img_path]


seed = 42
set_seed(seed)
data = CustomDataset()
loader = DataLoader(data, 128, shuffle=False)


# Initialize Img2Vec with GPU
models = ['resnet-18', 'alexnet', 'vgg', 'densenet'] #default: 'resnet-18'
img2vec = Img2Vec(cuda=False, model="efficientnet_b7")


unique_id = 0
img_emo_path_emb = [] # [elem, elem] elem: [bilde, emo, path, embeddings]
for batch in tqdm(loader):
  lst = [tnsr for tnsr in batch[0]]
  pils = [TF.to_pil_image(tensor.squeeze(0)) for tensor in lst]
  vectors = img2vec.get_vec(pils)
  vectors = torch.from_numpy(vectors)
  img_emo_path_emb.append([batch[0], batch[1][0], batch[1][1], vectors])

single_elements = [] #[elem, elem]. elem: [img, emo, path, embeddings]
for batch in img_emo_path_emb:
   print(type(batch[2]))
   for i in range(len(batch[0])):
      single_elements.append([batch[0][i], batch[1][i], batch[2][i], batch[3][i]])

distinct_emo = [0,1,2,3,4,5,6]
dicit = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: []}

def all_lists_have_ten_elements(d):
    return all(len(lst) == 100 for lst in d.values())


for one_img in single_elements: #appender slik at dict. har en fra hver emo-klasse
  if not all_lists_have_ten_elements(dicit):
    emo = int(one_img[1])
    if len(dicit[emo]) != 100:
      dicit[emo].append(one_img)
    else:
      continue
  else:
    break
  
print("lenth of one of them:", len(dicit[0]))


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
  def __init__(self, n_steps=1000, time_emb_dim=2560, context_dim=None): #switch time emb fra 512 til 100
        super(ConditionalUNet, self).__init__()
        # Sinusoidal embedding
        self.p_uncond = 0.2
        self.time_embed = nn.Embedding(n_steps, time_emb_dim) #embedding layer 1000 nedover, 100 bortover
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)
        self.label_enc = nn.Embedding(7, 2560)
        self.context_dim = context_dim
        self.time_emb_dim = 2560#100 + 512

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
        if label.shape == (16, 2560): #skjekke for emo-embeddings vs image-embeddings
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
  
######################################

model_path = 'ddpm_efficienet_ema.pt'
model = CondDDPM(ConditionalUNet(context_dim=2560), device=device)  # Replace with your model definition
state_dict = torch.load(model_path)
model.load_state_dict(state_dict)


def generate_new_images(ddpm, labels, n_samples=100, device=None, frames_per_gif=100, gif_name="sampling.gif", c=1, h=64, w=64):
    k = 3.0
    frame_idxs = np.linspace(0, ddpm.n_steps, frames_per_gif).astype(np.uint)
    frames = []

    with torch.no_grad():
        if device is None:
            device = ddpm.device


        x = torch.randn(n_samples, c, h, w).to(device)
        emo_label = labels.to(device)
        #embs =  #shape: ([16, 512])

        for idx, t in enumerate(list(range(ddpm.n_steps))[::-1]): #iterere fra denoising steps to 0, gjøre classifier-free sampling per gang

            time_tensor = (torch.ones(n_samples, 1, dtype=torch.long) * t).to(device)#.long()

            eta_theta_cond = ddpm.backward(x, time_tensor, emo_label) * (1 + k)
            eta_theta_uncond = ddpm.backward(x, time_tensor, torch.zeros_like(emo_label)) * k #torch.zeros(10, 1).to(device)
            eta_theta = eta_theta_cond - eta_theta_uncond

            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]

            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

            if t > 0:
                z = torch.randn(n_samples, c, h, w).to(device)


                beta_t = ddpm.betas[t]
                sigma_t = beta_t.sqrt()

                x = x + sigma_t * z


    return x


generated_lst = [] # liste med liste. hvert element: [ ]
for emo in dicit:
  img_lst = []
  emo_lst = []
  path_lst = []
  emb_lst = []

  for unique in dicit[emo]: #unique: [img, emo, path, embeddings]
    img_lst.append(unique[0])
    emo_lst.append(unique[1])
    path_lst.append(unique[2])
    emb_lst.append(unique[3].unsqueeze(0))        

  emb = torch.cat(emb_lst, dim=0).unsqueeze(1)
  print(emb.shape)
        #emb = emo[2].repeat(10,1).unsqueeze(1)
       # print(emb.shape)
  generated = generate_new_images(
  model,
  emb,
  n_samples=100,
  device=device,
  gif_name="fashion.gif")

  generated_lst.append([img_lst, emo_lst, path_lst,  generated])

torch.save(generated_lst, '1efficientnet_b7_generated_test.pt') #liste med følelser. hvert følelse: img_lst, emo_lst, path_lst, generated. skjekke typen til generated, men tror det er en tensor med 100,1,64,64
print("done trainimg, it was saved")
print("shape of emo class is:", generated_lst[0][1], generated_lst[0][1][0], generated_lst[0][1][0].shape)
print("done trainimg")



   