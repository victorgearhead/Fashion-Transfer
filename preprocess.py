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


def get_palette(num_cls):
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            palette[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            palette[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i += 1
            lab >>= 3
    return palette

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn

def save_output(image_name, pred, d_dir):
    predict = pred.squeeze()
    predict_np = predict.cpu().data.numpy()
    im = Image.fromarray(predict_np*255).convert('RGB')
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)
    os.makedirs(d_dir, exist_ok=True)
    imo.save(os.path.join(d_dir, f'saliency_map_inference.png'))

def process_single_image(image_path, device="cpu"):
    cloth_mask_dir = "output/cloth_mask"
    saliency_map_dir = "output/saliency_map"
    cloth_checkpoint_path = "mask/results/training_cloth_segm/checkpoints/mask_u2net.pth"
    saliency_model_path = "segmentation/saved_models/segm_u2net.pth"
    
    os.makedirs(cloth_mask_dir, exist_ok=True)
    os.makedirs(saliency_map_dir, exist_ok=True)
    
    transform_rgb = transforms.Compose([
        transforms.ToTensor(),
        Normalize_image(0.5, 0.5)
    ])
    
    net_mask = U2NET_MASK(in_ch=3, out_ch=4)
    net_mask = load_checkpoint_mgpu(net_mask, cloth_checkpoint_path)
    net_mask = net_mask.to(device).eval()
    
    img = Image.open(image_path).convert("RGB")
    image_tensor = transform_rgb(img)
    image_tensor = torch.unsqueeze(image_tensor, 0)
    
    with torch.no_grad():
        output_tensor = net_mask(image_tensor.to(device))
        output_tensor = F.log_softmax(output_tensor[0], dim=1)
        output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
    
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_arr = output_tensor.cpu().numpy()
    
    output_img = Image.fromarray(output_arr.astype("uint8"), mode="L")
    palette = get_palette(4)
    output_img.putpalette(palette)
    output_img.save(os.path.join(cloth_mask_dir, f'cloth_mask_inference.png'))
    
    net_saliency = U2NET(3, 1)
    if torch.cuda.is_available():
        net_saliency.load_state_dict(torch.load(saliency_model_path, weights_only=False))
        net_saliency.cuda()
    else:
        net_saliency.load_state_dict(torch.load(saliency_model_path, map_location='cpu', weights_only=False))
    net_saliency.eval()
    
    test_salobj_dataset = SalObjDataset(
        img_name_list=[image_path],
        lbl_name_list=[],
        transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])
    )
    
    test_salobj_dataloader = DataLoader(
        test_salobj_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    
    for data_test in test_salobj_dataloader:
        inputs_test = data_test['image'].type(torch.FloatTensor)
        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)
        
        d1, d2, d3, d4, d5, d6, d7 = net_saliency(inputs_test)
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)
        save_output(image_path, pred, saliency_map_dir)
        break

if __name__ == "__main__":
    image_name = "inference"
    image_path = f"input/images/inference.jpg"  # Change this to your image path
    process_single_image(image_path)