
#%% Importing modules
import cv2
import numpy as np
import requests
from requests.structures import CaseInsensitiveDict
import plotly.express as px

#%% Taking the video and extracting frames
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

video_path = r"Gameplay video"
x = 86  # Number of frames to extract
start_time = 4  # Start time in seconds
end_time = 90  # End time in seconds

images = extract_random_frames(video_path, x, start_time, end_time)

#%% Matching the features
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
# Estimate the intrinsic matrix
width, height = images[0].shape[1], images[0].shape[0]
focal_length = (width + height) / 2
intrinsic_matrix = np.array([[focal_length, 0, width / 2],
                             [0, focal_length, height / 2],
                             [0, 0, 1]])


def estimate_camera_poses(matched_features, intrinsic_matrix):
    camera_poses = [np.eye(4)]  # First camera pose is identity

    for match in matched_features:
        kp1 = np.float32([match['kp1'][m.queryIdx].pt for m in match['matches']])
        kp2 = np.float32([match['kp2'][m.trainIdx].pt for m in match['matches']])

        # Estimate the essential matrix
        E, mask = cv2.findEssentialMat(kp1, kp2, intrinsic_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)

        # Recover the relative camera pose
        _, R, t, _ = cv2.recoverPose(E, kp1, kp2, intrinsic_matrix)

        # Convert the rotation and translation matrices into a 4x4 homogeneous matrix
        relative_pose = np.eye(4)
        relative_pose[:3, :3] = R
        relative_pose[:3, 3] = t[:, 0]

        # Update the camera pose by multiplying the previous pose with the relative pose
        prev_pose = camera_poses[-1]
        current_pose = prev_pose @ relative_pose
        camera_poses.append(current_pose)

    return camera_poses

# Estimate camera poses
camera_poses = estimate_camera_poses(matched_features, intrinsic_matrix)

#%% Plot camera's position but interactive
import plotly.io as pio
import plotly.graph_objs as go
import plotly.express as px
pio.renderers.default = 'browser'

def plot_camera_trajectory(camera_poses):
    # Extract the camera positions
    camera_positions = [pose[:3, 3] for pose in camera_poses]

    # Create a 3D scatter plot for the camera positions
    x, y, z = zip(*camera_positions)
    scatter = go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=5, color='red'))

    # Connect the camera positions with lines
    lines = []
    for i in range(len(camera_positions) - 1):
        x_values = [camera_positions[i][0], camera_positions[i+1][0]]
        y_values = [camera_positions[i][1], camera_positions[i+1][1]]
        z_values = [camera_positions[i][2], camera_positions[i+1][2]]
        lines.append(go.Scatter3d(x=x_values, y=y_values, z=z_values, mode='lines', line=dict(width=2, color='blue')))

    # Combine scatter and lines into a single data list
    data = [scatter] + lines

    # Set the layout for the 3D plot
    layout = go.Layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )

    # Create and show the plot
    fig = go.Figure(data=data, layout=layout)
    fig.show()

# Call the function with the estimated camera poses
plot_camera_trajectory(camera_poses)
