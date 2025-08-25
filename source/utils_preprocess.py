# This file contains function for video or image collection preprocessing.
# For video we do the preprocessing and select k sharpest frames.
# Afterwards scene is constructed 
import cv2
import numpy as np
from tqdm import tqdm
import pycolmap
import os
import time
import tempfile
from moviepy import VideoFileClip
from matplotlib import pyplot as plt
from PIL import Image
import cv2
from tqdm import tqdm

WORKDIR = "../outputs/"


def get_rotation_moviepy(video_path):
    clip = VideoFileClip(video_path)
    rotation = 0

    try:
        displaymatrix = clip.reader.infos['inputs'][0]['streams'][2]['metadata'].get('displaymatrix', '')
        if 'rotation of' in displaymatrix:
            angle = float(displaymatrix.strip().split('rotation of')[-1].split('degrees')[0])
            rotation = int(angle) % 360
            
    except Exception as e:
        print(f"No displaymatrix rotation found: {e}")

    clip.reader.close()
    #if clip.audio:
    #    clip.audio.reader.close_proc()

    return rotation

def resize_max_side(frame, max_size):
    h, w = frame.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    return frame

def read_video_frames(video_input, k=1, max_size=1024):
    """
    Extracts every k-th frame from a video or list of images, resizes to max size, and returns frames as list.

    Parameters:
        video_input (str, file-like, or list): Path to video file, file-like object, or list of image files.
        k (int): Interval for frame extraction (every k-th frame).
        max_size (int): Maximum size for width or height after resizing.

    Returns:
        frames (list): List of resized frames (numpy arrays).
    """
    # Handle list of image files (not single video in a list)
    if isinstance(video_input, list):
        # If it's a single video in a list, treat it as video
        if len(video_input) == 1 and video_input[0].name.endswith(('.mp4', '.avi', '.mov')):
            video_input = video_input[0]  # unwrap single video file
        else:
            # Treat as list of images
            frames = []
            for img_file in video_input:
                img = Image.open(img_file.name).convert("RGB")
                img.thumbnail((max_size, max_size))
                frames.append(np.array(img)[...,::-1])
            return frames

    # Handle file-like or path
    if hasattr(video_input, 'name'):
        video_path = video_input.name
    elif isinstance(video_input, (str, os.PathLike)):
        video_path = str(video_input)
    else:
        raise ValueError("Unsupported video input type. Must be a filepath, file-like object, or list of images.")

    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video {video_path}.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    frames = []

    with tqdm(total=total_frames // k, desc="Processing Video", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % k == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = frame.shape[:2]
                scale = max(h, w) / max_size
                if scale > 1:
                    frame = cv2.resize(frame, (int(w / scale), int(h / scale)))
                frames.append(frame[...,[2,1,0]])
                pbar.update(1)
            frame_count += 1

    cap.release()
    return frames

def resize_max_side(frame, max_size):
    """
    Resizes the frame so that its largest side equals max_size, maintaining aspect ratio.
    """
    height, width = frame.shape[:2]
    max_dim = max(height, width)
    
    if max_dim <= max_size:
        return frame  # No need to resize

    scale = max_size / max_dim
    new_width = int(width * scale)
    new_height = int(height * scale)

    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_frame



def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()
    
def process_all_frames(IMG_FOLDER = '/scratch/datasets/hq_data/night2_all_frames',
                       to_visualize=False,
                       save_images=True):
    dict_scores = {}
    for idx, img_name in tqdm(enumerate(sorted([x for x in os.listdir(IMG_FOLDER) if '.png' in x]))):
        
        img = cv2.imread(os.path.join(IMG_FOLDER, img_name))#[250:, 100:]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fm = variance_of_laplacian(gray) + \
                variance_of_laplacian(cv2.resize(gray, (0,0), fx=0.75, fy=0.75)) + \
                variance_of_laplacian(cv2.resize(gray, (0,0), fx=0.5, fy=0.5)) + \
                variance_of_laplacian(cv2.resize(gray, (0,0), fx=0.25, fy=0.25))
        if to_visualize:
            plt.figure()
            plt.title(f"Laplacian score: {fm:.2f}")
            plt.imshow(img[..., [2,1,0]])
            plt.show()
        dict_scores[idx] = {"idx" : idx, 
                            "img_name" : img_name,
                            "score" : fm}
        if save_images:
            dict_scores[idx]["img"] = img
        
    return dict_scores

def select_optimal_frames(scores, k):
    """
    Selects a minimal subset of frames while ensuring no gaps exceed k.

    Args:
        scores (list of float): List of scores where index represents frame number.
        k (int): Maximum allowed gap between selected frames.

    Returns:
        list of int: Indices of selected frames.
    """
    n = len(scores)
    selected = [0, n-1]
    i = 0  # Start at the first frame

    while i < n:
        # Find the best frame to select within the next k frames
        best_idx = max(range(i, min(i + k + 1, n)), key=lambda x: scores[x], default=None)

        if best_idx is None:
            break  # No more frames left

        selected.append(best_idx)
        i = best_idx + k + 1  # Move forward, ensuring gaps stay within k

    return sorted(selected)


def variance_of_laplacian(image):
    """
    Compute the variance of Laplacian as a focus measure.
    """
    return cv2.Laplacian(image, cv2.CV_64F).var()

def preprocess_frames(frames, verbose=False):
    """
    Compute sharpness scores for a list of frames using multi-scale Laplacian variance.

    Args:
        frames (list of np.ndarray): List of frames (BGR images).
        verbose (bool): If True, print scores.

    Returns:
        list of float: Sharpness scores for each frame.
    """
    scores = []

    for idx, frame in enumerate(tqdm(frames, desc="Scoring frames")):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        fm = (
            variance_of_laplacian(gray) +
            variance_of_laplacian(cv2.resize(gray, (0, 0), fx=0.75, fy=0.75)) +
            variance_of_laplacian(cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)) +
            variance_of_laplacian(cv2.resize(gray, (0, 0), fx=0.25, fy=0.25))
        )
        
        if verbose:
            print(f"Frame {idx}: Sharpness Score = {fm:.2f}")

        scores.append(fm)

    return scores

def select_optimal_frames(scores, k):
    """
    Selects k frames by splitting into k segments and picking the sharpest frame from each.

    Args:
        scores (list of float): List of sharpness scores.
        k (int): Number of frames to select.

    Returns:
        list of int: Indices of selected frames.  
    """
    n = len(scores)
    selected_indices = []
    segment_size = n // k

    for i in range(k):
        start = i * segment_size
        end = (i + 1) * segment_size if i < k - 1 else n  # Last chunk may be larger
        segment_scores = scores[start:end]
        
        if len(segment_scores) == 0:
            continue  # Safety check if some segment is empty
        
        best_in_segment = start + np.argmax(segment_scores)
        selected_indices.append(best_in_segment)

    return sorted(selected_indices)

def save_frames_to_scene_dir(frames, scene_dir):
    """
    Saves a list of frames into the target scene directory under 'images/' subfolder.

    Args:
        frames (list of np.ndarray): List of frames (BGR images) to save.
        scene_dir (str): Target path where 'images/' subfolder will be created.
    """
    images_dir = os.path.join(scene_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    for idx, frame in enumerate(frames):
        filename = os.path.join(images_dir, f"{idx:08d}.png")  # 00000000.png, 00000001.png, etc.
        cv2.imwrite(filename, frame)

    print(f"Saved {len(frames)} frames to {images_dir}")


def run_colmap_on_scene(scene_dir):
    """
    Runs feature extraction, matching, and mapping on all images inside scene_dir/images using pycolmap.

    Args:
        scene_dir (str): Path to scene directory containing 'images' folder.
    
    TODO: if the function hasn't managed to match all the frames either increase image size,
    increase number of features or just remove those frames from the folder scene_dir/images
    """
    start_time = time.time()
    print(f"Running COLMAP pipeline on all images inside {scene_dir}")

    # Setup paths
    database_path = os.path.join(scene_dir, "database.db")
    sparse_path = os.path.join(scene_dir, "sparse")
    image_dir = os.path.join(scene_dir, "images")
    
    # Make sure output directories exist
    os.makedirs(sparse_path, exist_ok=True)

    # Step 1: Feature Extraction
    pycolmap.extract_features(
        database_path,
        image_dir,
        sift_options={
            "max_num_features": 512 * 2,
            "max_image_size": 512 * 1,
        }
    )
    print(f"Finished feature extraction in {(time.time() - start_time):.2f}s.")

    # Step 2: Feature Matching
    pycolmap.match_exhaustive(database_path)
    print(f"Finished feature matching in {(time.time() - start_time):.2f}s.")

    # Step 3: Mapping
    pipeline_options = pycolmap.IncrementalPipelineOptions()
    pipeline_options.min_num_matches = 15
    pipeline_options.multiple_models = True
    pipeline_options.max_num_models = 50
    pipeline_options.max_model_overlap = 20
    pipeline_options.min_model_size = 10
    pipeline_options.extract_colors = True
    pipeline_options.num_threads = 8
    pipeline_options.mapper.init_min_num_inliers = 30
    pipeline_options.mapper.init_max_error = 8.0
    pipeline_options.mapper.init_min_tri_angle = 5.0

    reconstruction = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=image_dir,
        output_path=sparse_path,
        options=pipeline_options,
    )
    print(f"Finished incremental mapping in {(time.time() - start_time):.2f}s.")

    # Step 4: Post-process Cameras to SIMPLE_PINHOLE
    recon_path = os.path.join(sparse_path, "0")
    reconstruction = pycolmap.Reconstruction(recon_path)

    for cam in reconstruction.cameras.values():
        cam.model = 'SIMPLE_PINHOLE'
        cam.params = cam.params[:3]  # Keep only [f, cx, cy]

    reconstruction.write(recon_path)

    print(f"Total pipeline time: {(time.time() - start_time):.2f}s.")

