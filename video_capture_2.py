# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 16:52:56 2023

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

    #frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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

#video_path = r"F:\Videos\World of Tanks\World of Tanks 2023.04.26 - 14.36.29.17.DVR.mp4"
video_path = r"F:\Videos\World of Tanks\World of Tanks 2023.05.01 - 11.48.30.26.mp4"
x = 10  # Number of frames to extract
start_time = 0  # Start time in seconds
end_time = 45  # End time in seconds

images = extract_random_frames(video_path, x, start_time, end_time)

#%%
def match_sift_features(images, min_match_distance=10):
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
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

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
            'des1': des1,  # Add this line
            'des2': des2,  # Add this line
            'matches': good_matches
        })

    return matched_features

matched_features = match_sift_features(images)

#%%
def nothing(x):
    pass

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        img1, img2, kp1, kp2, good_matches, scale = param

        for m in good_matches:
            pt1 = tuple(np.int32(kp1[m.queryIdx].pt))
            pt2 = tuple(np.int32(kp2[m.trainIdx].pt) + np.array([img1.shape[1], 0]))
            pt1_scaled = tuple(np.int32(np.array(pt1) * scale))
            pt2_scaled = tuple(np.int32(np.array(pt2) * scale))

            if np.linalg.norm(np.array(pt1_scaled) - np.array((x, y))) < 5:
                print(f"Match distance: {m.distance}")
                break
            elif np.linalg.norm(np.array(pt2_scaled) - np.array((x, y))) < 5:
                print(f"Match distance: {m.distance}")
                break

def update_matches(val, img1, img2, kp1, kp2, des1, des2, flann):
    ratio_threshold = cv2.getTrackbarPos('Ratio Threshold', 'Keypoints and Matches') / 100
    min_match_distance = cv2.getTrackbarPos('Min Match Distance', 'Keypoints and Matches')
    #min_keypoint_distance = cv2.getTrackbarPos('Min Keypoint Distance', 'Keypoints and Matches')

    #min_match_distance = 70
    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance and m.distance > min_match_distance:
            good_matches.append(m)

    display_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    display_img = cv2.resize(display_img, None, fx=0.5, fy=0.5)
    cv2.imshow('Keypoints and Matches', display_img)

def display_keypoints_and_matches(images, matched_features, pair_index):
    img1 = images[matched_features[pair_index]['img1']]
    img2 = images[matched_features[pair_index]['img2']]
    kp1 = matched_features[pair_index]['kp1']
    kp2 = matched_features[pair_index]['kp2']
    des1 = matched_features[pair_index]['des1']
    des2 = matched_features[pair_index]['des2']
    
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    windows_name = 'Keypoints and Matches'
    cv2.namedWindow(windows_name)
    cv2.createTrackbar('Ratio Threshold', windows_name, 70, 100, lambda val: update_matches(val, img1, img2, kp1, kp2, des1, des2, flann))
    cv2.createTrackbar('Min Match Distance', windows_name, 10, 100, lambda val: update_matches(val, img1, img2, kp1, kp2, des1, des2, flann))
    cv2.createTrackbar('Min Keypoint Distance', windows_name, 5, 100, lambda val: update_matches(val, img1, img2, kp1, kp2, des1, des2, flann))

    ratio_threshold = cv2.getTrackbarPos('Ratio Threshold', windows_name) / 100
    min_match_distance = cv2.getTrackbarPos('Min Match Distance', windows_name)
    min_keypoint_distance = cv2.getTrackbarPos('Min Keypoint Distance', windows_name)
    
    update_matches(None, img1, img2, kp1, kp2, des1, des2, flann)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage: Display keypoints and matches between the first and second images
display_keypoints_and_matches(images, matched_features, 2)

#%%
def remove_close_keypoints(keypoints, descriptors, min_distance=10):
    filtered_keypoints = []
    filtered_descriptors = []
    
    for idx, kp in enumerate(keypoints):
        is_too_close = False
        
        for existing_kp in filtered_keypoints:
            distance = np.linalg.norm(np.array(kp.pt) - np.array(existing_kp.pt))
            
            if distance < min_distance:
                is_too_close = True
                break
                
        if not is_too_close:
            filtered_keypoints.append(kp)
            filtered_descriptors.append(descriptors[idx])
            
    return filtered_keypoints, np.array(filtered_descriptors)


def update_matches(val, img1, img2, kp1, kp2, des1, des2, flann):
    ratio_threshold = cv2.getTrackbarPos('Ratio Threshold', 'Keypoints and Matches') / 100
    min_match_distance = cv2.getTrackbarPos('Min Match Distance', 'Keypoints and Matches')
    min_keypoint_distance = cv2.getTrackbarPos('Min Keypoint Distance', 'Keypoints and Matches')
    
    min_keypoint_distance = 10
    filtered_kp1, filtered_des1 = remove_close_keypoints(kp1, des1, min_keypoint_distance)
    filtered_kp2, filtered_des2 = remove_close_keypoints(kp2, des2, min_keypoint_distance)

    matches = flann.knnMatch(filtered_des1, filtered_des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance and m.distance > min_match_distance:
            good_matches.append(m)

    display_img = cv2.drawMatches(img1, filtered_kp1, img2, filtered_kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    display_img = cv2.resize(display_img, None, fx=0.5, fy=0.5)
    cv2.imshow('Keypoints and Matches', display_img)

def display_keypoints_and_matches(images, matched_features, pair_index):
    img1 = images[matched_features[pair_index]['img1']]
    img2 = images[matched_features[pair_index]['img2']]
    kp1 = matched_features[pair_index]['kp1']
    kp2 = matched_features[pair_index]['kp2']
    des1 = matched_features[pair_index]['des1']
    des2 = matched_features[pair_index]['des2']
    
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    cv2.namedWindow('Keypoints and Matches')
    cv2.createTrackbar('Ratio Threshold', 'Keypoints and Matches', 70, 100, lambda val: update_matches(val, img1, img2, kp1, kp2, des1, des2, flann))
    cv2.createTrackbar('Min Match Distance', 'Keypoints and Matches', 10, 100, lambda val: update_matches(val, img1, img2, kp1, kp2, des1, des2, flann))
    cv2.createTrackbar('Min Keypoint Distance', 'Keypoints and Matches', 5, 100, lambda val: update_matches(val, img1, img2, kp1, kp2, des1, des2, flann))
    
    update_matches(None, img1, img2, kp1, kp2, des1, des2, flann)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    


# Example usage: Display keypoints and matches between the first and second images
display_keypoints_and_matches(images, matched_features, 2)

import xvideos-dl

#%% Ça ça fonctionne, c'est principalement pour voir la distance entre les points
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        img1, img2, kp1, kp2, good_matches, scale = param

        for m in good_matches:
            pt1 = tuple(np.int32(kp1[m.queryIdx].pt))
            pt2 = tuple(np.int32(kp2[m.trainIdx].pt) + np.array([img1.shape[1], 0]))
            pt1_scaled = tuple(np.int32(np.array(pt1) * scale))
            pt2_scaled = tuple(np.int32(np.array(pt2) * scale))

            if np.linalg.norm(np.array(pt1_scaled) - np.array((x, y))) < 5:
                print(f"Match distance: {m.distance}")
                break
            elif np.linalg.norm(np.array(pt2_scaled) - np.array((x, y))) < 5:
                print(f"Match distance: {m.distance}")
                break

def update_matches(val, img1, img2, kp1, kp2, des1, des2, flann, callback=None):
    ratio_threshold = cv2.getTrackbarPos('Ratio Threshold', 'Keypoints and Matches') / 100
    min_match_distance = cv2.getTrackbarPos('Min Match Distance', 'Keypoints and Matches')

    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < ratio_threshold * n.distance and m.distance > min_match_distance]

    display_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    scale = 0.5
    display_img = cv2.resize(display_img, None, fx=scale, fy=scale)
    cv2.imshow('Keypoints and Matches', display_img)

    if callback:
        cv2.setMouseCallback('Keypoints and Matches', callback, (img1, img2, kp1, kp2, good_matches, scale))

def display_keypoints_and_matches(images, matched_features, pair_index):
    img1 = images[matched_features[pair_index]['img1']]
    img2 = images[matched_features[pair_index]['img2']]
    kp1 = matched_features[pair_index]['kp1']
    kp2 = matched_features[pair_index]['kp2']
    des1 = matched_features[pair_index]['des1']
    des2 = matched_features[pair_index]['des2']

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    windows_name = 'Keypoints and Matches'
    cv2.namedWindow(windows_name)
    cv2.createTrackbar('Ratio Threshold', windows_name, 57, 100, lambda val: update_matches(val, img1, img2, kp1, kp2, des1, des2, flann, mouse_callback))
    cv2.createTrackbar('Min Match Distance', windows_name, 48, 160, lambda val: update_matches(val, img1, img2, kp1, kp2, des1, des2, flann, mouse_callback))
    update_matches(None, img1, img2, kp1, kp2, des1, des2, flann)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage: Display keypoints and matches between the first and second images
display_keypoints_and_matches(images, matched_features, 3)

#%%
def apply_mask(image, mask):
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image

def remove_hud_elements(images):
    fgbg = cv2.createBackgroundSubtractorMOG2()

    # Apply background subtraction to all images to create masks
    masks = []
    for img in images:
        fgmask = fgbg.apply(img)
        masks.append(fgmask)

    # Combine all masks to create a single mask for the HUD elements
    combined_mask = np.zeros_like(masks[0])
    for mask in masks:
        combined_mask = cv2.add(combined_mask, mask)

    # Threshold the combined mask to create a binary mask
    _, thresh_mask = cv2.threshold(combined_mask, 127, 255, cv2.THRESH_BINARY)

    # Apply the binary mask to the images to remove HUD elements
    masked_images = [apply_mask(img, thresh_mask) for img in images]

    return masked_images, combined_mask

# Example usage:
masked_images, combined_mask = remove_hud_elements(images)

def overlay_mask(image, mask, alpha=0.5, color=(0, 0, 255)):
    # Invert the mask
    inverted_mask = cv2.bitwise_not(mask)
    
    # Create a color version of the mask
    colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    colored_mask[np.where((colored_mask == [255, 255, 255]).all(axis=2))] = color

    # Overlay the color mask on the image
    overlayed_image = cv2.addWeighted(image, 1, colored_mask, alpha, 0)

    return overlayed_image

def display_mask_overlay(image, mask):
    overlayed_image = overlay_mask(image, mask)

    cv2.imshow("Mask Overlay", overlayed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage:
image_index = 5  # Select the index of the image you want to visualize
selected_image = images[image_index]
selected_mask = combined_mask  # Assuming 'masks' is the list of masks from the remove_hud_elements function

display_mask_overlay(selected_image, selected_mask)
