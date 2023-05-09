# -*- coding: utf-8 -*-
"""
Created on Fri May  5 15:15:16 2023

@author: jfturpin
"""
#%% Importing modules
import cv2
import numpy as np
from typing import List
import torch
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
import supervision as sv

#%%
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

video_path = r"..."
x = 4  # Number of frames to extract
start_time = 0  # Start time in seconds
end_time = 120  # End time in seconds

images = extract_random_frames(video_path, x, start_time, end_time)

#%%

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = r".\groundingdino\config\GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH =  r".\GroundingDINO\weights\groundingdino_swint_ogc.pth"

# Building GroundingDINO inference model
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

CLASSES = ['tank']
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

for image in images:
    # detect objects
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=CLASSES,
        box_threshold=BOX_THRESHOLD,
        text_threshold=BOX_THRESHOLD
    )
    
    # annotate image with detections
    box_annotator = sv.BoxAnnotator()
    labels = [
        f"{CLASSES[class_id]} {confidence:0.2f}" 
        for _, _, confidence, class_id, _ 
        in detections]
    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
    
    # save the annotated grounding dino image
    
    # cv2.imwrite(IMAGE_WRITE, annotated_frame)
    
    cv2.imshow("groundingdino_annotated_image", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#%%
import pickle

# Create a NumPy array to be pickled
arr = np.array([1, 2, 3, 4, 5])

#Pickle the array to disk
# with open(r"array.pkl", 'wb') as f:
#     pickle.dump(detections, f)

# Load the pickled array back into memory
with open('array.pkl', 'rb') as f:
    detections = pickle.load(f)

#print(xyxy) # Output: [1 2 3 4 5]


#%%
SAM_ENCODER_VERSION = "vit_b"
SAM_CHECKPOINT_PATH = r"H:\envs\torch_2\GroundingDINO\Grounded-Segment-Anything\sam_vit_b_01ec64.pth"

# Building SAM Model and SAM Predictor
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
sam.to(device='cuda')
sam_predictor = SamPredictor(sam)



# Prompting SAM with detected boxes
def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


# convert detections to masks
detections.mask = segment(
    sam_predictor=sam_predictor,
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
    xyxy=detections.xyxy
)

# annotate image with detections
box_annotator = sv.BoxAnnotator()
mask_annotator = sv.MaskAnnotator()
labels = [
    f"{CLASSES[class_id]} {confidence:0.2f}" 
    for _, _, confidence, class_id, _ 
    in detections]
annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

# save the annotated grounded-sam image
cv2.imshow("grounded_sam_annotated_image", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
