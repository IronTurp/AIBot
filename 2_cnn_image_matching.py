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

class ImageSearchEngine:
    def __init__(self, image_dir):
        # Initialize the device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained model
        self.model = models.resnet50(pretrained=True).to(self.device)
        self.model.eval()

        # Prepare image transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Extract features from images
        self.image_dir = image_dir
        self.image_files = glob.glob(self.image_dir + '/*.jpg') + glob.glob(self.image_dir + '/*.png')

        # Test the model to get the dimension before creating the faiss index
        img_path = self.image_files[0]
        features = self.extract_features(img_path)
        model_dimension = features.shape[1]

        # Create the index
        self.index = faiss.IndexFlatL2(model_dimension)  # Assume the output of the model is 1000 dimension

        # Extract the features for each image and add to index
        for image_file in self.image_files:
            features = self.extract_features(os.path.join(self.image_dir, image_file))
            self.index.add(features)
            
        # Number of images to extract
        self.n = 3

    def extract_features(self, img_path):
        img = Image.open(img_path)
        t_img = self.transform(img)
        t_img = t_img.unsqueeze(0).to(self.device)  # Put the tensor on the GPU
        features = self.model(t_img)
        return features.cpu().detach().numpy()  # Move tensor back to CPU for numpy compatibility

    def search(self, new_image_path):
        features = self.extract_features(new_image_path)
        D, I = self.index.search(features, self.n)  # Retrieve top 3 most similar images
        return [(self.image_files[I[0][i]], D[0][i]) for i in range(self.n)]  # Return the filenames and similarities

# Use the search function
engine = ImageSearchEngine(path_to_images)
new_image = path_to_test_image
results = engine.search(new_image)
for most_similar_image, similarity in results:
    print('Most similar image is ', most_similar_image, ' with similarity ', similarity)
