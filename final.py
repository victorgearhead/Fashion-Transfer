from PIL import Image
import os
import numpy as np
import sys
from tqdm import tqdm
from skimage import io
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import glob
from torchvision.models import vgg19, VGG19_Weights
import copy
from mask.data.base_dataset import Normalize_image
from mask.utils.checkpoints import load_checkpoint_mgpu

from mask.network.u2net import U2NET_MASK

from segmentation.data_loader import RescaleT
from segmentation.data_loader import ToTensor
from segmentation.data_loader import ToTensorLab
from segmentation.data_loader import SalObjDataset

from segmentation.model import U2NET

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

imsize = 512 if torch.cuda.is_available() else 128

loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor()])

def image_loader(image_name):
    image = Image.open(image_name)
    original_size = image.size
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float), original_size

style_img, _ = image_loader("./input/style/style.png")
content_img, original_size = image_loader(f"./input/images/inference.jpg")
    
assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"
unloader = transforms.ToPILImage()

class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input
    
def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input
cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    normalization = Normalization(normalization_mean, normalization_std)
    content_losses = []
    style_losses = []
    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses
input_img = content_img.clone()

def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img])
    return optimizer
def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=100000, content_weight=1):
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    input_img.requires_grad_(True)
    model.eval()
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)
    run = [0]
    while run[0] <= num_steps:
        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1

            return style_score + content_score

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img
output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img)


resized_output = F.interpolate(output, size=(original_size[1], original_size[0]), mode='bilinear', align_corners=False)
final_output = unloader(resized_output.squeeze(0))

final_output.save(f"output/total_style/final_output_inference.jpg")

loader = transforms.Compose([
    transforms.ToTensor()])

mask = Image.open(f"output/cloth_mask/cloth_mask_inference.png").convert("L")
mask_tensor = transforms.ToTensor()(mask)

binary_mask = (mask_tensor > 0.1).float()

if binary_mask.ndim == 3:
    binary_mask = binary_mask[:1, :, :]  
binary_mask = binary_mask.unsqueeze(0)

def image_loader(image_name):
    image = Image.open(image_name)
    original_size = image.size
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float), original_size

styled_img = loader(final_output).unsqueeze(0)

binary_mask = binary_mask.to(device, torch.float)
binary_mask = binary_mask.to(device)
styled_img = styled_img.to(device)

final_img = binary_mask * styled_img

unloader = transforms.ToPILImage()
final_output = unloader(final_img.squeeze(0))

mask = Image.open(f"output/saliency_map/saliency_map_inference.png").convert("L")
cloth_grayscale = final_output.convert("L")
mask_tensor = transforms.ToTensor()(mask)
cloth_grayscale_tensor = transforms.ToTensor()(cloth_grayscale)

binary_mask = (mask_tensor > 0.1).float()
binary_cloth_mask = (cloth_grayscale_tensor > 0.1).float()

if binary_mask.ndim == 3:
    binary_mask = binary_mask[:1, :, :]  
binary_mask = binary_mask.unsqueeze(0)

if binary_cloth_mask.ndim == 3:
    binary_cloth_mask = binary_cloth_mask[:1, :, :]  
binary_cloth_mask = binary_cloth_mask.unsqueeze(0)

def image_loader(image_name):
    image = Image.open(image_name)
    original_size = image.size
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float), original_size

content_img, _ = image_loader(f"./input/images/inference.jpg")
styled_cloth = loader(final_output).unsqueeze(0)

binary_mask = binary_mask.to(device, torch.float)
binary_cloth_mask = binary_cloth_mask.to(device, torch.float)

binary_mask = binary_mask.to(device)
binary_cloth_mask = binary_cloth_mask.to(device)
styled_cloth = styled_cloth.to(device)
content_img = content_img.to(device)

result = (binary_mask * styled_cloth + (1 - binary_mask) * content_img) + content_img * binary_mask * (1 - binary_cloth_mask)

unloader = transforms.ToPILImage()
final_output = unloader(result.squeeze(0))
final_output.save(f"output/final_outputs/final_inference.jpg")