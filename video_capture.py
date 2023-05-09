# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 14:39:33 2023

@author: jfturpin
"""


#%% Importing modules
import cv2
import numpy as np
import requests
from requests.structures import CaseInsensitiveDict
import plotly.express as px
import open3d as o3d

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

#video_path = r"F:\Videos\World of Tanks\World of Tanks 2023.04.26 - 14.36.29.17.DVR.mp4"
video_path = r"F:\Videos\World of Tanks\World of Tanks 2023.05.01 - 11.48.30.26.mp4"
video_path = r"F:\Videos\World of Tanks\World of Tanks 2023.05.01 - 12.49.48.27.mp4"
x = 86  # Number of frames to extract
start_time = 4  # Start time in seconds
end_time = 90  # End time in seconds

images = extract_random_frames(video_path, x, start_time, end_time)

#%% Querying the server to process the depth maps from the images
def get_depth_maps(image):
    url = 'https://model-zoo.metademolab.com/predictions/dino_depth'

    headers = CaseInsensitiveDict()
    headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/112.0'
    headers['Accept'] = 'application/json, text/plain, */*'
    headers['Accept-Language'] = 'en-US,en;q=0.5'
    headers['Accept-Encoding'] = 'gzip, deflate, br'
    headers['Origin'] = 'https://dinov2.metademolab.com'
    headers['Connection'] = 'keep-alive'
    headers['Referer'] = 'https://dinov2.metademolab.com/'
    headers['Sec-Fetch-Dest'] = 'empty'
    headers['Sec-Fetch-Mode'] = 'cors'
    headers['Sec-Fetch-Site'] = 'same-site'
    headers['DNT'] = '1'
    headers['Sec-GPC'] = '1'
    headers['Pragma'] = 'no-cache'
    headers['Cache-Control'] = 'no-cache'
    
    ret, png_data = cv2.imencode('.png', image)
    if not ret:
        print("Error encoding image to PNG format")
        return None

    files = {'data': ('blob', png_data.tobytes(), 'image/png')}
    response = requests.post(url, headers=headers, files=files)
    
    if response.status_code == 200:
        # Save the returned image
        #with open('output_image.png', 'wb') as f:
            #f.write(response.content)
        print('Depth map saved as output_image.png')
        return response.content
    else:
        print(f'Request failed with status code: {response.status_code}')
        return False

depth_maps = []
for image in images:
    depth_maps.append(get_depth_maps(image))

#%% Matching the features
def match_sift_features(images, min_match_distance=10):
    # Initialize the SIFT feature detector
    sift = cv2.SIFT_create()

    # Initialize the feature matcher (FLANN-based)
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matched_features = []
    
    ratio_threshold = 0.57
    min_match_distance = 48

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
            if m.distance < ratio_threshold * n.distance and m.distance > min_match_distance:
                #good_matches = [m for m, n in matches if m.distance < ratio_threshold * n.distance and m.distance > min_match_distance]
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

#%% Plot the camera's estimated movement Static
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_camera_trajectory(camera_poses):
    # Extract the camera positions
    camera_positions = [pose[:3, 3] for pose in camera_poses]

    # Create a 3D scatter plot for the camera positions
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = zip(*camera_positions)
    ax.scatter(x, y, z, c='r', marker='o')

    # Connect the camera positions with lines
    for i in range(len(camera_positions) - 1):
        x_values = [camera_positions[i][0], camera_positions[i+1][0]]
        y_values = [camera_positions[i][1], camera_positions[i+1][1]]
        z_values = [camera_positions[i][2], camera_positions[i+1][2]]
        ax.plot(x_values, y_values, z_values, c='b')

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()

# Call the function with the estimated camera poses
plot_camera_trajectory(camera_poses)

#%%Here I want to adjust the coordinates, I know that between pose 0 and 1, the movement in the y axis is null
#This is just for visualization purposes.

# def adjust_camera_poses(camera_poses, frame_to_adjust):
#     adjusted_poses = camera_poses.copy()

#     # Modify the Y-axis translation of the camera pose between frame_to_adjust and frame_to_adjust + 1
#     adjusted_poses[frame_to_adjust + 1][:3, 3][1] = adjusted_poses[frame_to_adjust][:3, 3][1]

#     # Recalculate the camera poses from the adjusted pose
#     for i in range(frame_to_adjust + 1, len(camera_poses) - 1):
#         relative_pose = np.linalg.inv(adjusted_poses[i]) @ adjusted_poses[i + 1]
#         adjusted_poses[i + 1] = adjusted_poses[i] @ relative_pose

#     return adjusted_poses

# # Adjust the camera poses
# adjusted_camera_poses = adjust_camera_poses(camera_poses, 0)

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

#%%
def triangulate_points(matched_features, camera_poses, intrinsic_matrix):
    all_3d_points = []

    for i, match in enumerate(matched_features):
        kp1 = np.float32([match['kp1'][m.queryIdx].pt for m in match['matches']])
        kp2 = np.float32([match['kp2'][m.trainIdx].pt for m in match['matches']])

        pose1 = camera_poses[i]
        pose2 = camera_poses[i + 1]

        # Calculate projection matrices for both camera poses
        P1 = intrinsic_matrix @ pose1[:3, :]
        P2 = intrinsic_matrix @ pose2[:3, :]

        # Triangulate points
        points_3d_homogeneous = cv2.triangulatePoints(P1, P2, kp1.T, kp2.T)
        points_3d = points_3d_homogeneous[:3, :] / points_3d_homogeneous[3, :]

        all_3d_points.append(points_3d.T)

    # Combine all 3D points into a single array
    point_cloud = np.vstack(all_3d_points)

    return point_cloud

def create_open3d_point_cloud(point_cloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    return pcd

# Triangulate points and create a point cloud
point_cloud = triangulate_points(matched_features, camera_poses, intrinsic_matrix)
pcd = create_open3d_point_cloud(point_cloud)

def remove_outliers(pcd, nb_neighbors=20, std_ratio=2.0):
    # Remove outliers using statistical outlier removal method
    processed_pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return processed_pcd

# Remove outliers from the point cloud
pcd_filtered = remove_outliers(pcd)

def create_axis(size=1.0):
    axis = o3d.geometry.TriangleMesh()

    # X-axis (red)
    cylinder_x = o3d.geometry.TriangleMesh.create_cylinder(radius=0.005 * size, height=size)
    cylinder_x.paint_uniform_color([1, 0, 0])
    cylinder_x.translate([size / 2, 0, 0])
    cylinder_x.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0, np.pi / 2, 0)))
    axis += cylinder_x

    # Y-axis (green)
    cylinder_y = o3d.geometry.TriangleMesh.create_cylinder(radius=0.005 * size, height=size)
    cylinder_y.paint_uniform_color([0, 1, 0])
    cylinder_y.translate([0, size / 2, 0])
    axis += cylinder_y

    # Z-axis (blue)
    cylinder_z = o3d.geometry.TriangleMesh.create_cylinder(radius=0.005 * size, height=size)
    cylinder_z.paint_uniform_color([0, 0, 1])
    cylinder_z.translate([0, 0, size / 2])
    cylinder_z.rotate(o3d.geometry.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0)))
    axis += cylinder_z

    return axis


def visualize_point_cloud(pcd):
    axis = create_axis()
    o3d.visualization.draw_geometries([pcd], point_show_normal=True)

# Visualize the point cloud
visualize_point_cloud(pcd_filtered)

#%%
def create_mesh_from_point_cloud(pcd, depth=8, scale=1.1, linear_fit=True):
    # Estimate normals for the point cloud
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # Apply Poisson surface reconstruction
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth, scale=scale, linear_fit=linear_fit)
    return mesh

# Create a mesh from the point cloud
mesh = create_mesh_from_point_cloud(pcd_filtered)

# Visualize the mesh along with the point cloud and the axes
axis = create_axis()
o3d.visualization.draw_geometries([pcd_filtered, mesh, axis], point_show_normal=True)

#%%
def triangulate_features(matched_features, camera_poses, intrinsic_matrix):
    point_clouds = []

    # Iterate through matched features
    for match in matched_features:
        img1_idx = match['img1']
        img2_idx = match['img2']
        kp1 = match['kp1']
        kp2 = match['kp2']
        matches = match['matches']

        # Get the camera poses (extrinsic matrices) for the image pair
        pose1 = camera_poses[img1_idx]
        pose2 = camera_poses[img2_idx]

        # Calculate the projection matrices for each camera
        P1 = intrinsic_matrix @ pose1[:3]
        P2 = intrinsic_matrix @ pose2[:3]

        # Extract the matched keypoints' coordinates
        points1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        points2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        # Triangulate the matched keypoints
        points_4d = cv2.triangulatePoints(P1, P2, points1.T, points2.T)

        # Convert homogeneous coordinates to 3D points
        points_3d = points_4d[:3] / points_4d[3]

        # Store the 3D points for this pair of images
        point_clouds.append(points_3d)

    return point_clouds

point_clouds = triangulate_features(matched_features, camera_poses, intrinsic_matrix)

merged_point_cloud = np.hstack(point_clouds)

import plotly.express as px

#%%
import matplotlib.pyplot as plt

def display_keypoints_and_matches(images, matched_features, pair_index):
    # Get the image pair, keypoints, and matches
    img1 = images[matched_features[pair_index]['img1']]
    img2 = images[matched_features[pair_index]['img2']]
    kp1 = matched_features[pair_index]['kp1']
    kp2 = matched_features[pair_index]['kp2']
    matches = matched_features[pair_index]['matches']

    # Draw the keypoints and matches on the images
    display_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Convert the image to RGB for displaying with matplotlib
    display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)

    cv2.imshow('Matched SIFT Keypoints', display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
    # # Display the images with keypoints and matches
    # plt.figure(figsize=(16, 8))
    # plt.imshow(display_img)
    # plt.axis('off')
    # plt.show()

# Example usage: Display keypoints and matches between the first and second images
display_keypoints_and_matches(images, matched_features, 0)


#%% Plot the cloud in 3d
def plot_point_cloud(point_cloud):
    x, y, z = point_cloud

    # Create a scatter plot of the 3D points
    fig = px.scatter_3d(x=x, y=y, z=z, opacity=0.5)

    # Customize the appearance of the plot
    fig.update_traces(marker=dict(size=2, line=dict(width=0)))
    fig.update_layout(scene=dict(xaxis_title='X',
                                  yaxis_title='Y',
                                  zaxis_title='Z',
                                  aspectmode='data'))

    # Show the plot
    fig.show()

plot_point_cloud(merged_point_cloud)

#%%
import plotly.io as pio

def save_point_cloud_html(point_cloud, filename):
    x, y, z = point_cloud

    # Create a scatter plot of the 3D points
    fig = px.scatter_3d(x=x, y=y, z=z, opacity=0.5)

    # Customize the appearance of the plot
    fig.update_traces(marker=dict(size=2, line=dict(width=0)))
    fig.update_layout(scene=dict(xaxis_title='X',
                                  yaxis_title='Y',
                                  zaxis_title='Z',
                                  aspectmode='data'))

    # Save the plot as an HTML file
    pio.write_html(fig, file=filename)

save_point_cloud_html(merged_point_cloud, 'point_cloud.html')



    #%%
    # Convert the response content to a list of lists
    depth_map_list = json.loads(response.content)

    # Convert the list of lists to a 2D NumPy array
    depth_map_array = np.array(depth_map_list)

    # Normalize the array to 8-bit (0-255) values
    depth_map_normalized = ((depth_map_array - depth_map_array.min()) * (1 / (depth_map_array.max() - depth_map_array.min()) * 255)).astype('uint8')

    # Create a PIL Image object from the NumPy array
    depth_map_image = Image.fromarray(depth_map_normalized)

    # Display the depth map image
    depth_map_image.show()
    
#%%


def update_matches(val, img1, img2, kp1, kp2, des1, des2, flann):
    ratio_threshold = cv2.getTrackbarPos('Ratio Threshold', 'Keypoints and Matches') / 100
    min_match_distance = cv2.getTrackbarPos('Min Match Distance', 'Keypoints and Matches')
    min_keypoint_distance = cv2.getTrackbarPos('Min Keypoint Distance', 'Keypoints and Matches')

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

    cv2.namedWindow('Keypoints and Matches')
    cv2.createTrackbar('Ratio Threshold', 'Keypoints and Matches', 70, 100, lambda val: update_matches(val, img1, img2, kp1, kp2, des1, des2, flann))
    cv2.createTrackbar('Min Match Distance', 'Keypoints and Matches', 10, 100, lambda val: update_matches(val, img1, img2, kp1, kp2, des1, des2, flann))
    cv2.createTrackbar('Min Keypoint Distance', 'Keypoints and Matches', 5, 100, lambda val: update_matches(val, img1, img2, kp1, kp2, des1, des2, flann))
    min_keypoint_distance = cv2.getTrackbarPos('Min Keypoint Distance', 'Keypoints and Matches')
    
    update_matches(None, img1, img2, kp1, kp2, des1, des2, flann)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage: Display keypoints and matches between the first and second images
display_keypoints_and_matches(images, matched_features, 2)

#%%

def update_matches(val, img1, img2, kp1, kp2, des1, des2, flann):
    ratio_threshold = cv2.getTrackbarPos('Ratio Threshold', 'Keypoints and Matches') / 100
    min_match_distance = cv2.getTrackbarPos('Min Match Distance', 'Keypoints and Matches')
    min_keypoint_distance = cv2.getTrackbarPos('Min Keypoint Distance', 'Keypoints and Matches')
    
    
    #filtered_kp1, filtered_des1 = remove_close_keypoints(kp1, des1, min_keypoint_distance)
    #filtered_kp2, filtered_des2 = remove_close_keypoints(kp2, des2, min_keypoint_distance)

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

# Example usage: Display keypoints and matches between the first and second images
display_keypoints_and_matches(images, matched_features, 2)

#%%
def on_trackbar_change(val, params):
    img1, img2, kp1, kp2, des1, des2, flann = params
    
    ratio_threshold = cv2.getTrackbarPos('Ratio Threshold', 'Keypoints and Matches') / 100
    min_match_distance = cv2.getTrackbarPos('Min Match Distance', 'Keypoints and Matches')
    min_keypoint_distance = cv2.getTrackbarPos('Min Keypoint Distance', 'Keypoints and Matches')

    kp1_filtered, des1_filtered = remove_close_keypoints(kp1, des1, min_keypoint_distance)
    kp2_filtered, des2_filtered = remove_close_keypoints(kp2, des2, min_keypoint_distance)

    update_matches(val, img1, img2, kp1_filtered, kp2_filtered, des1_filtered, des2_filtered, flann)



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

    filtered_kp1 = remove_close_keypoints(kp1, min_keypoint_distance)
    filtered_kp2 = remove_close_keypoints(kp2, min_keypoint_distance)

    matches = flann.knnMatch(des1, des2, k=2)
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
    callback_params = (img1, img2, kp1, kp2, des1, des2, flann)
    cv2.createTrackbar('Ratio Threshold', 'Keypoints and Matches', 70, 100, lambda val: on_trackbar_change(val, callback_params))
    cv2.createTrackbar('Min Match Distance', 'Keypoints and Matches', 10, 100, lambda val: on_trackbar_change(val, callback_params))
    cv2.createTrackbar('Min Keypoint Distance', 'Keypoints and Matches', 10, 100, lambda val: on_trackbar_change(val, callback_params))

    update_matches(None, img1, img2, kp1, kp2, des1, des2, flann)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage: Display keypoints and matches between the first and second images
display_keypoints_and_matches(images, matched_features, 1)
