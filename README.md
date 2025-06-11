---
title: Fashion Style Transfer
app: https://huggingface.co/spaces/VictorGearhead/Fashion-Style-Transfer
sdk: streamlit
sdk_version: 1.44.1
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
# Style Transfer in Fashion Images

This project implements a deep learning-based system for style transfer in fashion images, combining cloth segmentation, saliency map generation, and neural style transfer to create a virtual try-on experience. The system stylizes clothing regions in an input image while preserving the background, using saliency-based blending for seamless integration.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Model Weights](#model-weights)
- [Results](#results)
- [References](#references)
- [Team Members](#team-members)

## Project Overview
The project leverages deep learning techniques to apply artistic styles to clothing in images, enabling users to visualize how garments would look with different textures or patterns. It consists of three main components:
1. **Cloth Segmentation and Saliency Map Generation**: Uses U-2-Net models to segment clothing regions and generate saliency maps highlighting prominent areas.
2. **Neural Style Transfer**: Applies VGG-19 to transfer artistic styles to the segmented clothing regions.
3. **Saliency-Based Blending**: Combines the stylized clothing with the original image using a saliency map for natural transitions.

## Features
- Precise cloth segmentation using U-2-Net with four output classes (background, upper-body, lower-body, full-body clothing).
- Saliency map generation to focus on prominent clothing regions.
- Neural style transfer using VGG-19 for high-quality stylization.
- Seamless blending of stylized clothing with the original image.
- Support for both CPU and GPU processing.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd fashion-transfer-main
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the pre-trained model weights from the provided [Google Drive link](https://drive.google.com/drive/folders/1FswqwdYqUvuS_kFIk-BnBK2nBh2MYrAY?usp=sharing) and place them in the appropriate directories:
   - Cloth segmentation model: `mask/results/training_cloth_segm/checkpoints/mask_u2net.pth`
   - Saliency model: `segmentation/saved_models/segm_u2net.pth`

## Usage
1. Prepare input images:
   - Place the input image in `input/images/` (e.g., `inference.jpg`).
   - Place the style image in `input/style/` (e.g., `style.png`).
2. Run the preprocessing script to generate cloth masks and saliency maps:
   ```bash
   python preprocessing.py
   ```
   This generates:
   - Cloth mask: `output/cloth_mask/cloth_mask_inference.png`
   - Saliency map: `output/saliency_map/saliency_map_inference.png`
3. Run the style transfer and blending script:
   ```bash
   python final.py
   ```
   This generates the final stylized image in `output/final_outputs/final_inference.jpg`.

## Project Structure
```
├── input/
│   ├── images/                # Input images (e.g., inference.jpg)
│   └── style/                 # Style images (e.g., style.png)
├── output/
│   ├── cloth_mask/            # Generated cloth masks
│   ├── saliency_map/          # Generated saliency maps
│   └── final_outputs/         # Final stylized images
├── mask/
│   ├── data/                  # Dataset utilities
│   ├── network/               # U-2-Net model for cloth segmentation
│   └── utils/                 # Checkpoint loading utilities
├── segmentation/
│   ├── saved_models/          # Pre-trained saliency model
│   └── data_loader.py         # Data loading utilities
├── preprocessing.py           # Script for cloth segmentation and saliency map generation
├── final.py                   # Script for style transfer and blending
├── report.pdf                 # Project report
└── README.md                  # This file
```

## Dependencies
- Python 3.8+
- PyTorch
- torchvision
- PIL (Pillow)
- NumPy
- scikit-image
- tqdm

Install dependencies using:
```bash
pip install torch torchvision pillow numpy scikit-image tqdm
```

## Model Weights
Pre-trained model weights are required for cloth segmentation and saliency map generation. Download them from:
[Google Drive](https://drive.google.com/drive/folders/1FswqwdYqUvuS_kFIk-BnBK2nBh2MYrAY?usp=sharing)

Place the weights in:
- `mask/results/training_cloth_segm/checkpoints/mask_u2net.pth`
- `segmentation/saved_models/segm_u2net.pth`

## Results
The system produces high-quality stylized images with natural transitions between the stylized clothing and the original background. The saliency-based blending ensures clear clothing boundaries, making the results suitable for virtual try-on applications and fashion design previews.

## References
1. U-2-Net Model: [Paper](https://arxiv.org/abs/2005.09007), [Code](https://github.com/xuebinqin/U-2-Net)
2. Dataset: [iMaterialist Fashion 2019](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/data)
3. Neural Style Transfer: [PyTorch Tutorial](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)
4. Saliency-Guided Image Style Transfer: [IEEE Paper](https://ieeexplore.ieee.org/abstract/document/8794904)

## Team Members
- Chittiprolu Bhala Vignesh (B22AI015)
- Singamsetti Manikanta Varshit (B22AI038)
- Atharva Date (B22AI045)
