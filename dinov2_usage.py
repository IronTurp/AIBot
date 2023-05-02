# -*- coding: utf-8 -*-
"""
Created on Tue May  2 13:54:10 2023

@author: jeanfrancois.turpin
"""

#DINO V2 LOSS
import torch
#import torchvision
from torchvision import transforms
import torch.nn as nn
import cv2
import numpy as np

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda().to(DEVICE)
model.eval()

def resize_image(image, patch_size=14):
    height, width, _ = image.shape

    # Calculate new dimensions
    new_height = int(np.ceil(height / patch_size) * patch_size)
    new_width = int(np.ceil(width / patch_size) * patch_size)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))

    return resized_image

# Read the image
image_path = r""
image = cv2.imread(image_path)

# Resize the image
patch_size = 14
resized_image = resize_image(image, patch_size)

transform = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
img_tensor = transform(resized_image).unsqueeze(0)

# Pass the input image through the model and get the predicted mask
with torch.no_grad():
    mask_tensor = model(img_tensor.cuda())

# Convert the mask tensor to a numpy array and apply a threshold to get the final mask
mask_np = mask_tensor[0].argmax(dim=0).cpu().numpy()
