import torch
import torch.nn as nn
from torchvision.models import resnet18

class CustomModel(nn.Module):
    def __init__(self, num_key_classes, num_mouse_classes):
        super(CustomModel, self).__init__()
        
        # Load the pretrained ResNet18 model and remove the last layer to use it as a feature extractor
        self.feature_extractor = resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])
        
        # Define the classifier for keyboard key presses
        self.keyboard_classifier = nn.Sequential(
            nn.Linear(512, 256),  # Linear layer with 512 input features and 256 output features
            nn.ReLU(),           # ReLU activation function
            nn.Linear(256, num_key_classes)  # Linear layer with 256 input features and output features equal to the number of key classes
        )
        
        # Define the classifier for mouse movements
        self.mouse_classifier = nn.Sequential(
            nn.Linear(512 + 4, 256),  # Linear layer with 516 input features (512 from feature_extractor + 4 from mouse_data) and 256 output features
            nn.ReLU(),                # ReLU activation function
            nn.Linear(256, num_mouse_classes)  # Linear layer with 256 input features and output features equal to the number of mouse classes
        )
    
    def forward(self, image, mouse_data):
        # Pass the input image through the feature extractor
        features = self.feature_extractor(image).view(-1, 512)
        
        # Use the extracted features as input for the keyboard key press classifier
        keyboard_output = self.keyboard_classifier(features)
        
        # Concatenate the extracted features with the mouse_data and use it as input for the mouse movement classifier
        mouse_output = self.mouse_classifier(torch.cat((features, mouse_data), dim=1))
        
        return keyboard_output, mouse_output
