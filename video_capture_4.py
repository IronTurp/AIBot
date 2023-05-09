# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 08:37:07 2023

@author: jfturpin
"""

#%% Importing modules
import cv2
import numpy as np

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

video_path = r"F:\Videos\World of Tanks\World of Tanks 2023.04.26 - 14.36.29.17.DVR.mp4"
x = 10  # Number of frames to extract
start_time = 0  # Start time in seconds
end_time = 120  # End time in seconds

images = extract_random_frames(video_path, x, start_time, end_time)

#%%

def load_images_and_keypoints(images):
    """
    Loads images and computes SIFT keypoints and descriptors for each image.
    
    :param num_images: int, number of images to load
    :return: tuple, (images, keypoints, descriptors)
             images: dict, keys are image indices and values are images
             keypoints: dict, keys are image indices and values are lists of keypoints
             descriptors: dict, keys are image indices and values are lists of descriptors
    """
    keypoints = {}
    descriptors = {}
    sift = cv2.SIFT_create()

    for i in range(len(images) - 1):        
        kp, des = sift.detectAndCompute(images[i], None)        
        keypoints[i] = kp
        descriptors[i] = des

    return keypoints, descriptors

def find_similar_keypoints(keypoints, descriptors, distance_ratio_threshold=0.75, location_threshold=10):
    """
    Finds keypoints with similar descriptors and locations across all images.
    
    :param keypoints: dict, keys are image indices and values are lists of keypoints
    :param descriptors: dict, keys are image indices and values are lists of descriptors
    :param distance_ratio_threshold: float, threshold for descriptor similarity (default: 0.75)
    :param location_threshold: float, threshold for keypoint location similarity (default: 10)
    :return: set, similar keypoints found across images
    """
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
    """
    Displays an image with similar keypoints overlaid.
    
    :param image: ndarray, the image to display
    :param similar_keypoints: iterable, keypoints to overlay on the image
    """
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
