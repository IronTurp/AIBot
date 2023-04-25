import torch
import torch.nn as nn
from torchvision.models import resnet18

class CustomModel(nn.Module):
    def __init__(self, num_key_classes):
        super(CustomModel, self).__init__()
        
        # Define the feature extractor (ResNet18) and remove the last layer
        self.feature_extractor = resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])
        
        # Define the combined representation layer
        # It takes the concatenated image features (512), mouse data (4), and one-hot encoded keyboard data
        # and reduces the dimensionality to 256
        self.combined_representation = nn.Sequential(
            nn.Linear(512 + 4 + num_key_classes, 256),
            nn.ReLU()
        )
        
        # Define the keyboard classifier
        # It takes the 256-dimensional joint representation and predicts the keyboard key press probabilities
        self.keyboard_classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_key_classes)
        )
        
        # Define the mouse regression network
        # It takes the 256-dimensional joint representation and predicts the continuous values
        # for the change in x and y coordinates of the mouse movement
        self.mouse_regression = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # The output size is now 2 (x and y coordinate changes)
        )
    
    def forward(self, image, mouse_data, keyboard_data):
        # Pass the input image through the feature extractor to obtain a 512-dimensional feature vector
        features = self.feature_extractor(image).view(-1, 512)
        
        # Concatenate the 512-dimensional feature vector, mouse data, and one-hot encoded keyboard data
        # Pass the concatenated vector through the combined_representation layer
        combined = self.combined_representation(torch.cat((features, mouse_data, keyboard_data), dim=1))
        
        # Pass the 256-dimensional joint representation through the keyboard classifier
        # to obtain the keyboard key press probabilities
        keyboard_output = self.keyboard_classifier(combined)
        
        # Pass the 256-dimensional joint representation through the mouse regression network
        # to obtain the predicted change in x and y coordinates of the mouse movement
        mouse_output = self.mouse_regression(combined)
        
        # Return the keyboard key press probabilities and the mouse movement (delta_x, delta_y)
        return keyboard_output, mouse_output
