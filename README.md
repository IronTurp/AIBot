# AIBot
Quick prototype of a generalized AI bot

main.py
The code defines a custom PyTorch model called CustomModel. This model has two parts: a feature extractor and two classifiers. The feature extractor is based on a pretrained ResNet18 model, and the classifiers are designed for predicting keyboard key presses and mouse movements.

The two classifiers (for keyboard and mouse) are separate, but they both take features extracted from the image by the feature extractor as input. In the forward method, the image is passed through the feature extractor, and the resulting features are used as input for both classifiers. The classifiers then output the predicted keyboard key presses and mouse movements.

The feature extractor in this case is based on a pretrained ResNet18 model, which is a deep convolutional neural network (CNN) architecture used for image recognition tasks. Unlike SIFT, which is a handcrafted feature extraction algorithm, the ResNet model learns to extract relevant features from the input images during training. The pretrained ResNet18 model has already been trained on a large dataset (e.g., ImageNet) and can extract meaningful features from the input images, making it useful for transfer learning.

nn.Sequential is a PyTorch container that chains multiple layers or modules together in a sequential manner. In the code, nn.Sequential is used to create two separate classifier networks, one for keyboard key presses and another for mouse movements. Each classifier consists of two linear layers with a ReLU activation function in between. These classifiers take the features extracted from the input images and produce predictions for keyboard and mouse events.
