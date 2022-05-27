'''
Tutorial Source: https://pytorch.org/vision/main/_modules/torchvision/transforms/functional.html

Importing the necessary libraries.
- timm: PyTorch Unofficial Image Models Lib
- request: Python library for making HTTP requests

Used torch version:
'''

from PIL import Image
import torch
import timm
import requests
import torchvision.transforms as transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

print(torch.__version__)

'''
Model Repo: https://github.com/facebookresearch/deit
'''


model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
model.eval()
print(model)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])

img_link = "https://n1.sdlcdn.com/imgs/g/o/f/iPhone-Black-iPhone-4s-16GB-SDL411082062-1-dfc7d.jpg"
img = Image.open(requests.get(img_link, stream=True).raw)

'''
Acceptable image tensor vars for inference are [Batch, Channel, Height, Width]
Since we're not inferring on a batch, we can just use the first element of the tensor.
Therefore we should put ```[None,]```
'''

img = transform(img)[None,]
print(img.shape)
out = model(img)

clsidx = torch.argmax(out)

label_link = "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt"
label = requests.get(label_link).text.split("\n")[clsidx]
print(label)