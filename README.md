# AIBot
Quick prototype of a generalized AI bot

After seeing a video of someone creating a simple, yet surprisingly effective, Fall Guys bot, I decided to try my hand at it.
This repo is, for the moment, a collection of semi-random spaghetti code.
If you find something useful, then great, otherwise, apologies.

After carefully analyzing the problem and trying multiple avenues, here is what I think has to be done:
1) Collecting screenshots from the game. If the game has multiple levels, collect images for every level and store them in separate folder.
2) For each level, use COLMAP to create a 3D reconstruction of the level and to extract the SIFT keypoints and descriptors, this is key.
3) Now that each level is processed, we have to find a way to let the bot know in real time where it is in the environment. This took me the longest to figure out the best approach. In 10 years, it'll be easy, because we'll be able to just brute force the SIFT features matching between the real-time gameplay, and the offline database. But for now, we have to find ways around it.
4) For example, when COLMAP registers a new image in the 3D reconstruction, it uses a vocabulary tree to first find the images most closely matching the new one.
5) Then it can infer where it was taken based on the other images and the 2D to 3D mapping that it created.
6) In my case, this is not practical since COLMAP will store the image and takes a long time to do the actual matching, I need to do it on fly.
7) Instead I use ResNet50 CNN to extract feature, and then use FAISS from META (Facebook Research is pretty darn impressive if you ask me, open sourcing their stuff like that, kudos to the Zuck for letting them do that, I'm impressed) to match the images.
8) Use the precalculated features, extract features from new image.
9) Do the matching
10) Calculate position in World.
11) And now, implement game logic.
12) Use something like YOLOv8 or YOLO-NAS for managing the shooting.

1) 1_colmap_cli_prepare.py: This code will prepare the colmap database
2) 2_cnn_image_matching.py: Using ResNet, we match the most similar image.
3) 3_feature_matching.py: With the new image and it's closest match, we match the features between them
4) 4_calculate_position.py: %TODO: calculate the estimate positions of the tank

*main.py*

The code defines a custom PyTorch model called CustomModel. This model has two parts: a feature extractor and two classifiers. The feature extractor is based on a pretrained ResNet18 model, and the classifiers are designed for predicting keyboard key presses and mouse movements.

The two classifiers (for keyboard and mouse) are separate, but they both take features extracted from the image by the feature extractor as input. In the forward method, the image is passed through the feature extractor, and the resulting features are used as input for both classifiers. The classifiers then output the predicted keyboard key presses and mouse movements.

The feature extractor in this case is based on a pretrained ResNet18 model, which is a deep convolutional neural network (CNN) architecture used for image recognition tasks. Unlike SIFT, which is a handcrafted feature extraction algorithm, the ResNet model learns to extract relevant features from the input images during training. The pretrained ResNet18 model has already been trained on a large dataset (e.g., ImageNet) and can extract meaningful features from the input images, making it useful for transfer learning.

nn.Sequential is a PyTorch container that chains multiple layers or modules together in a sequential manner. In the code, nn.Sequential is used to create two separate classifier networks, one for keyboard key presses and another for mouse movements. Each classifier consists of two linear layers with a ReLU activation function in between. These classifiers take the features extracted from the input images and produce predictions for keyboard and mouse events.
