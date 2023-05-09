# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 08:37:07 2023

@author: jfturpin
"""

#%% Importing modules
import cv2
import numpy as np
import requests
from requests.structures import CaseInsensitiveDict
import plotly.express as px

def extract_random_frames(video_path, x, start_time, end_time):
    """
    Extracts x frames from a video file between start_time and end_time.

    :param video_path: str, path to the video file
    :param x: int, number of frames to extract
    :param start_time: int, start time in seconds
    :param end_time: int, end time in seconds
    :return: list, list of extracted frames (images)
    """
    cap = cv2.VideoCapture(video_path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    start_frame = int(start_time * frame_rate)
    end_frame = int(end_time * frame_rate)
    total_frames = end_frame - start_frame

    # Calculate the step size for selecting frames
    step = total_frames // x

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    images = []
    for i in range(x):
        ret, frame = cap.read()
        if ret:
            images.append(frame)
        
        # Skip (step - 1) frames to maintain the required distance between selected frames
        for _ in range(step - 1):
            cap.read()

    cap.release()

    return images

video_path = r"F:\Videos\World of Tanks\World of Tanks 2023.04.26 - 14.36.29.17.DVR.mp4"
x = 10  # Number of frames to extract
start_time = 0  # Start time in seconds
end_time = 120  # End time in seconds

images = extract_random_frames(video_path, x, start_time, end_time)

#%%
def match_sift_features(images, min_match_distance=10, mask=None):
    """
    Matches SIFT features between consecutive images in the given list.

    :param images: list, list of images to match features
    :param min_match_distance: int, minimum match distance for a match to be considered good (default is 10)
    :param mask: numpy.ndarray or None, optional mask to be applied when detecting keypoints (default is None)
    :return: list, list of dictionaries containing matched features information for each pair of consecutive images
    """
    # Initialize the SIFT feature detector
    sift = cv2.SIFT_create()

    # Initialize the feature matcher (FLANN-based)
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matched_features = []

    # Iterate through pairs of images
    for i in range(len(images) - 1):
        img1 = images[i]
        img2 = images[i + 1]

        # Detect and compute SIFT features for each image
        kp1, des1 = sift.detectAndCompute(img1, mask)
        kp2, des2 = sift.detectAndCompute(img2, mask)

        # Match features between the image pair
        matches = flann.knnMatch(des1, des2, k=2)

        # Ratio test to filter good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance and m.distance > min_match_distance:
                good_matches.append(m)

        # Store the keypoints and good matches for the image pair
        matched_features.append({
            'img1': i,
            'img2': i + 1,
            'kp1': kp1,
            'kp2': kp2,
            'des1': des1,
            'des2': des2,
            'matches': good_matches
        })

    return matched_features

matched_features = match_sift_features(images, 10)

#%% Create the mask
def create_hud_mask(images, threshold=30, kernel_size=3):
    """
    Creates a mask representing the HUD area by analyzing frame differences in consecutive frames.

    :param images: list, list of images to create the HUD mask from
    :param threshold: int, threshold value used to create a binary mask (default is 30)
    :param kernel_size: int, size of the kernel used for morphological operations (default is 3)
    :return: numpy.ndarray, mask representing the HUD area
    """
    num_frames = len(images)
    frame_shape = images[0].shape[:2]  # Only take the height and width dimensions
    diff_sum = np.zeros(frame_shape, dtype=np.float32)

    # Calculate the absolute differences between consecutive frames
    for i in range(num_frames - 1):
        gray1 = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(images[i + 1], cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        diff_sum += diff

    # Compute the average difference
    diff_avg = diff_sum / (num_frames - 1)

    # Threshold the average difference to create a binary mask
    _, mask = cv2.threshold(diff_avg, threshold, 255, cv2.THRESH_BINARY)

    # Convert the mask to the proper data type
    mask = mask.astype(np.uint8)

    # Refine the mask with morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)

    return mask

def visualize_hud_mask(images, mask, alpha=0.5, selected_indices=None):
    """
    Visualizes the HUD mask on selected images.

    :param images: list, list of images to visualize the HUD mask on
    :param mask: numpy.ndarray, mask representing the HUD area
    :param alpha: float, transparency of the mask overlay (default is 0.5)
    :param selected_indices: list, indices of the images to visualize the mask on (default is None, which selects all images)
    """
    # If no indices are provided, display the mask on all images
    if selected_indices is None:
        selected_indices = range(len(images))

    # Iterate through the selected images
    for index in selected_indices:
        image = images[index].copy()
        overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # Convert the mask to BGR format to match the image
        image_with_mask = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)  # Overlay the mask on the image

        # Display the result using OpenCV
        window_name = f'Image with HUD Mask (Index: {index})'
        cv2.imshow(window_name, image_with_mask)
        cv2.waitKey(0)  # Wait for a key press
        cv2.destroyWindow(window_name)  # Close the window after a key is pressed

mask = create_hud_mask(images)
visualize_hud_mask(images, mask, selected_indices=None)
#visualize_hud_mask(images, mask, selected_indices=[0, 10, 20])

#%%
def filter_keypoints(images, threshold_ratio=0.9, min_match_count=10):
    """
    Filters keypoints detected by the SIFT algorithm in a list of images by removing those located in the same area in at least 90% of the images.
    
    :param images: list, list of images
    :param threshold_ratio: float, ratio of images in which a keypoint must appear in the same area to be considered for removal (default is 0.9)
    :param min_match_count: int, minimum number of matches required to consider two keypoints as a match (default is 10)
    
    :return: dict, dictionary containing the filtered keypoints for each image, with image indices as keys and lists of keypoints as values
    """
    # Initialize the SIFT detector
    sift = cv2.SIFT_create()

    # Calculate SIFT keypoints for each image and store them in a dictionary
    keypoints_dict = {}
    for idx, image in enumerate(images):
        keypoints, _ = sift.detectAndCompute(image, None)
        keypoints_dict[idx] = keypoints

    # Create a list of dictionaries containing matched feature information
    matches_list = []
    for idx in range(len(images) - 1):
        # Match keypoints using FLANN matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        _, des1 = sift.detectAndCompute(images[idx], None)
        _, des2 = sift.detectAndCompute(images[idx + 1], None)
        matches = flann.knnMatch(des1, des2, k=2)

        # Store the good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        if len(good_matches) >= min_match_count:
            matches_list.append({"matches": good_matches, "index1": idx, "index2": idx + 1})

    # Inspect the extracted keypoints and compare them with each other across all images
    keypoint_occurrences = {}
    for match_info in matches_list:
        for match in match_info["matches"]:
            img1_idx = match_info["index1"]
            img2_idx = match_info["index2"]
            kp1_idx = match.queryIdx
            kp2_idx = match.trainIdx
            keypoint_occurrences[(img1_idx, kp1_idx)] = keypoint_occurrences.get((img1_idx, kp1_idx), 0) + 1
            keypoint_occurrences[(img2_idx, kp2_idx)] = keypoint_occurrences.get((img2_idx, kp2_idx), 0) + 1

    # Remove keypoints that are located in the same area in at least 90% of the images
    num_images = len(images)
    threshold_count = int(threshold_ratio * num_images)
    filtered_keypoints = {}
    for img_idx, keypoints in keypoints_dict.items():
        filtered_keypoints[img_idx] = [
            keypoint for idx, keypoint in enumerate(keypoints) if keypoint_occurrences.get((img_idx, idx), 0) < threshold_count
        ]

    return filtered_keypoints

# Example usage
filtered_keypoints = filter_keypoints(images)

#%%
def visualize_filtered_keypoints(images, keypoints_dict, filtered_keypoints, selected_index):
    """
    Visualizes the keypoints on a selected image, marking excluded keypoints in red and included keypoints in green.

    :param images: list, list of images
    :param keypoints_dict: dict, dictionary containing original keypoints for each image
    :param filtered_keypoints: dict, dictionary containing filtered keypoints for each image
    :param selected_index: int, index of the selected image
    """
    image = images[selected_index].copy()

    # Create sets of keypoint locations for original and filtered keypoints
    original_locations = set((kp.pt[0], kp.pt[1]) for kp in keypoints_dict[selected_index])
    filtered_locations = set((kp.pt[0], kp.pt[1]) for kp in filtered_keypoints[selected_index])

    # Draw excluded keypoints in red
    for x, y in original_locations - filtered_locations:
        cv2.circle(image, (int(x), int(y)), 3, (0, 0, 255), -1)

    # Draw included keypoints in green
    for x, y in filtered_locations:
        cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)

    # Display the result using OpenCV
    window_name = f'Filtered Keypoints (Index: {selected_index})'
    cv2.imshow(window_name, image)
    cv2.waitKey(0)  # Wait for a key press
    cv2.destroyWindow(window_name)  # Close the window after a key is pressed
    
# Example usage
original_keypoints = {idx: keypoints for idx, keypoints in enumerate(images)}
selected_index = 1
visualize_filtered_keypoints(images, original_keypoints, filtered_keypoints, selected_index)

#%%

def calculate_keypoints(images):
    sift = cv2.SIFT_create()
    keypoints_dict = {}
    for idx, image in enumerate(images):
        keypoints, _ = sift.detectAndCompute(image, None)
        keypoints_dict[idx] = keypoints
    return keypoints_dict

def match_features(images, keypoints_dict):
    sift = cv2.SIFT_create()
    matches_list = []
    for idx in range(len(images) - 1):
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        _, des1 = sift.detectAndCompute(images[idx], None)
        _, des2 = sift.detectAndCompute(images[idx + 1], None)
        matches = flann.knnMatch(des1, des2, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        matches_list.append({"matches": good_matches, "index1": idx, "index2": idx + 1})
    return matches_list

def filter_keypoints(keypoints_dict, matches_list, threshold_ratio=0.9):
    keypoint_occurrences = {}
    for match_info in matches_list:
        for match in match_info["matches"]:
            img1_idx = match_info["index1"]
            img2_idx = match_info["index2"]
            kp1_idx = match.queryIdx
            kp2_idx = match.trainIdx
            keypoint_occurrences[(img1_idx, kp1_idx)] = keypoint_occurrences.get((img1_idx, kp1_idx), 0) + 1
            keypoint_occurrences[(img2_idx, kp2_idx)] = keypoint_occurrences.get((img2_idx, kp2_idx), 0) + 1

    num_images = len(keypoints_dict)
    threshold_count = int(threshold_ratio * num_images)
    filtered_keypoints = {}
    for img_idx, keypoints in keypoints_dict.items():
        filtered_keypoints[img_idx] = [
            keypoint for idx, keypoint in enumerate(keypoints) if keypoint_occurrences.get((img_idx, idx), 0) < threshold_count
        ]
    return filtered_keypoints

def visualize_keypoints(images, keypoints_dict, filtered_keypoints, selected_index):
    image = images[selected_index].copy()
    original_locations = set((kp.pt[0], kp.pt[1]) for kp in keypoints_dict[selected_index])
    filtered_locations = set((kp.pt[0], kp.pt[1]) for kp in filtered_keypoints[selected_index])

    for x, y in original_locations - filtered_locations:
        cv2.circle(image, (int(x), int(y)), 3, (0, 0, 255), -1)

    for x, y in filtered_locations:
        cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)

    window_name = f'Filtered Keypoints (Index: {selected_index})'
    cv2.imshow(window_name, image)
    cv2.waitKey(0)  # Wait for a key press
    cv2.destroyWindow(window_name)  # Close the window after a key is pressed

# Calculate keypoints for each image
keypoints_dict = calculate_keypoints(images)

# Match features between consecutive images
matches_list = match_features(images, keypoints_dict)

# Filter keypoints that are located in the same area in at least 90% of the images
filtered_keypoints = filter_keypoints(keypoints_dict, matches_list)

# Visualize the keypoints for a selected image (example: index 0)
selected_index = 2
visualize_keypoints(images, keypoints_dict, filtered_keypoints, selected_index)

#%%

def load_images_and_keypoints(num_images):
    images = {}
    keypoints = {}
    sift = cv2.SIFT_create()

    for i in range(num_images):
        img = cv2.imread(f'image_{i}.png')
        kp, _ = sift.detectAndCompute(img, None)
        images[i] = img
        keypoints[i] = kp

    return images, keypoints

def filter_hud_keypoints(keypoints, distance_threshold=10):
    num_images = len(keypoints)
    hud_keypoints = set()

    for i in range(num_images):
        for j in range(i+1, num_images):
            kp1, kp2 = keypoints[i], keypoints[j]
            for k1 in kp1:
                for k2 in kp2:
                    if np.linalg.norm(np.array(k1.pt) - np.array(k2.pt)) < distance_threshold:
                        hud_keypoints.add(k1)
                        hud_keypoints.add(k2)

    return hud_keypoints

def display_keypoints(images, keypoints, hud_keypoints):
    for idx, img in images.items():
        img_kp = cv2.drawKeypoints(img, list(set(keypoints[idx]) - hud_keypoints), None, color=(0, 255, 0))
        img_kp = cv2.drawKeypoints(img_kp, list(hud_keypoints), None, color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow(f'Image {idx}', img_kp)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    num_images = 10
    images, keypoints = load_images_and_keypoints(num_images)
    hud_keypoints = filter_hud_keypoints(keypoints)
    display_keypoints(images, keypoints, hud_keypoints)

if __name__ == '__main__':
    main()
    
filtered_keypoints = filter_hud_keypoints(keypoints_dict)

#%% Fonctionne

def load_images_and_keypoints(images):
    keypoints = {}
    descriptors = {}
    sift = cv2.SIFT_create()

    for i in range(len(images) - 1):        
        kp, des = sift.detectAndCompute(images[i], None)        
        keypoints[i] = kp
        descriptors[i] = des

    return keypoints, descriptors

def find_similar_keypoints(keypoints, descriptors, distance_ratio_threshold=0.75, location_threshold=10):
    num_images = len(keypoints)
    similar_keypoints = set()

    for i in range(num_images):
        for j in range(i+1, num_images):
            kp1, des1 = keypoints[i], descriptors[i]
            kp2, des2 = keypoints[j], descriptors[j]

            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)

            for m, n in matches:
                if m.distance < distance_ratio_threshold * n.distance:
                    k1, k2 = kp1[m.queryIdx], kp2[m.trainIdx]
                    if np.linalg.norm(np.array(k1.pt) - np.array(k2.pt)) < location_threshold:
                        similar_keypoints.add(k1)
                        similar_keypoints.add(k2)

    return similar_keypoints

def display_similar_keypoints(image, similar_keypoints):
    img_kp = cv2.drawKeypoints(image, list(similar_keypoints), None, color=(0, 255, 0))
    cv2.imshow('Similar Keypoints', img_kp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    keypoints, descriptors = load_images_and_keypoints(images)
    similar_keypoints = find_similar_keypoints(keypoints, descriptors)
    display_similar_keypoints(images[3], similar_keypoints)

if __name__ == '__main__':
    main()

