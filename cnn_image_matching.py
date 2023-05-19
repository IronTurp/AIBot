# -*- coding: utf-8 -*-
"""
Created on Thu May 18 12:30:19 2023

@author: jfturpin
"""

import torch
import torchvision.models as models
from PIL import Image
from torchvision import transforms
import faiss
import numpy as np
import os
import glob
import cv2
import time

# 1. Feature extraction

# Load pre-trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet50(pretrained=True).to(device)
model.eval()

# Prepare image transformation pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to extract features
def extract_features(model, img_path):
    img = Image.open(img_path)
    t_img = transform(img)
    t_img = t_img.unsqueeze(0).to(device)  # Put the tensor on the GPU
    features = model(t_img)
    return features.cpu().detach().numpy()  # Move tensor back to CPU for numpy compatibility


# Extract features from images
image_dir = r"H:\maps\Mines\images"
image_files = glob.glob(image_dir + '/*.jpg') + glob.glob(image_dir + '/*.png')

#Output of the model test
img_path = image_files[0]
features = extract_features(model, image_files[0])
index = faiss.IndexFlatL2(1000)  # Assume the output of the model is 1000 dimension

for image_file in image_files:
    features = extract_features(model, os.path.join(image_dir, image_file))
    index.add(features)

# 2. Indexing done in above loop

# 3. Search

def search(new_image_path):
    features = extract_features(model, new_image_path)
    D, I = index.search(features, 1)  # Retrieve top 1 most similar image
    return image_files[I[0][0]], D[0][0]  # Return the filename and similarity

# Use the search function
new_image = r"H:\maps\Mines\shot_045.jpg"
most_similar_image, similarity = search(new_image)
print('Most similar image is ', most_similar_image, ' with similarity ', similarity)

# Create SIFT object
sift = cv2.SIFT_create()

img = cv2.imread(most_similar_image)
# Resize the image
#img = cv2.resize(img, (512,512))

# Detect SIFT features, with no masks
keypoints, descriptors = sift.detectAndCompute(img, None)

#%%
n_executions = 10
execution_times = []

for _ in range(n_executions):
    start_time = time.time()
    
    # Detect SIFT features, with no masks
    keypoints, descriptors = sift.detectAndCompute(img, None)
    
    end_time = time.time()
    execution_times.append(end_time - start_time)

average_execution_time = sum(execution_times) / n_executions

print(f'Average execution time over {n_executions} executions: {average_execution_time} seconds')

