import torch
from torchvision import transforms
from PIL import Image
from img2vec_pytorch import Img2Vec
from PIL import Image
import torchvision.transforms.functional as TF
import glob
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

###fant her
#### from here: https://medium.com/analytics-vidhya/read-fer2013-face-expression-recognition-dataset-using-pytorch-torchvision-9ff64f55018e
##and here: https://medium.com/analytics-vidhya/creating-a-custom-dataset-and-dataloader-in-pytorch-76f210a1df5d


class CustomDataset(Dataset):
    def __init__(self):
        self.imgs_path = "fer2013/train/"
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
        return img_tensor, class_id


data = CustomDataset()
loader = DataLoader(data, 128, shuffle=True)


# Initialize Img2Vec with GPU
models = ['resnet-18', 'alexnet', 'vgg', 'densenet'] #default: 'resnet-18'
img2vec = Img2Vec(cuda=True)


unique_id = 0
embeddings = [] 
images = [] # [[tensor, class], [tensor, class], [ (60,1,64,64), (60,1) ]]
for batch in loader:
  lst = [tnsr for tnsr in batch[0]]
  pils = [TF.to_pil_image(tensor.squeeze(0)) for tensor in lst]
  vectors = img2vec.get_vec(pils)
  vectors = torch.from_numpy(vectors)
  images.append([batch[0], batch[1]])
  embeddings.append(vectors)

torch.save(images, 'images_64_fer2_resnet.pt')
torch.save(embeddings, 'embeddings_64_fer2_resnet.pt')
print(images[0][0].shape, images[0][1].shape,  embeddings[0].shape, len(images))

print("saved, shape:")



