import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import io
import re
import base64

__version__ = "0.1.0"
BASE_DIR = Path(__file__).resolve(strict=True).parent


class ConvolutionNNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionNNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        #############################
        self.drop_out = nn.Dropout(0.25)
        #############################
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(4096, 1000)

        #############################
        self.drop_out = nn.Dropout(0.5)
        #############################
        self.fc2 = nn.Linear(1000, 29)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


with open(f"{BASE_DIR}/arlett_net_model.ckpt", "rb") as f:
    model = ConvolutionNNetwork()
    checkpoint = torch.load(f)
    state_dict = model.state_dict()
    for k1, k2 in zip(state_dict.keys(), checkpoint.keys()):
        state_dict[k1] = checkpoint[k2]
    model.load_state_dict(state_dict)

classes = ['ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'لا', 'م', 'ن', 'ه', 'و', 'ي']

def decode_img(msg):
    msg = base64.b64decode(msg)
    buf = io.BytesIO(msg)
    img = Image.open(buf)
    return img

def pre_image(img):
    # checkpoint = torch.load(modelpath)
    # state_dict = model.state_dict()
    # for k1, k2 in zip(state_dict.keys(), checkpoint.keys()):
    #     state_dict[k1] = checkpoint[k2]
    # model.load_state_dict(state_dict)

    img = decode_img(img)
    imgnp = np.asarray(img)
    if(imgnp.shape[0] > 1):
        if(imgnp.shape[0] > 3):
            img = img.convert('RGB')

        transform_norm = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), transforms.Grayscale()])
    else:
        transform_norm = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    
    img_normalized = transform_norm(img)


    img_normalized = img_normalized.unsqueeze_(0)

    with torch.no_grad():
        model.eval()  
        output = model(img_normalized)
        index = output.data.cpu().numpy().argmax()
        probs = torch.softmax(output, dim=1)
        probs = probs[0][index].tolist()
        probs = round(probs, 2)

        class_name = classes[index]
        return class_name, probs


        