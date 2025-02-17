import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import shutil
import os
import re
import cv2
import ffmpeg
from PIL import Image
import yaml
import json
plt.style.use("https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle")
# Utils.py
import sympy
import subprocess
import shutil
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
    
# For videos
def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

        
def merge_yolo_datasets(dataset_dir_1, dataset_dir_2, output_dir, map_1=None, map_2=None):
    
    # Create output directories
    output_dir = os.path.join(output_dir, 'combined_dataset')
    os.makedirs(output_dir, exist_ok=True)
    
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
    
    for dataset_index, (dataset_dir, map_dict) in enumerate([(dataset_dir_1, map_1), (dataset_dir_2, map_2)], 1):
        for split in ['train', 'valid', 'test']:
            img_dir = os.path.join(dataset_dir, split, 'images')
            labels_dir = os.path.join(dataset_dir, split, 'labels')
            
            if os.path.exists(img_dir) and os.path.exists(labels_dir):
                # Copy images
                for img in os.listdir(img_dir):
                    shutil.copy2(os.path.join(img_dir, img), 
                                 os.path.join(output_dir, split, 'images', img))
                
                # Copy and potentially modify labels
                for label in os.listdir(labels_dir):
                    src_path = os.path.join(labels_dir, label)
                    dst_path = os.path.join(output_dir, split, 'labels', label)
                    
                    # Read label content
                    with open(src_path, 'r') as f:
                        lines = f.readlines()
                    
                    modified_lines = []
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            if map_dict and class_id in map_dict:
                                parts[0] = str(map_dict[class_id])
                            modified_lines.append(' '.join(parts) + '\n')
                    
                    # Write modified content
                    with open(dst_path, 'w') as f:
                        f.writelines(modified_lines)
            
            else:
                print(f"Dataset {dataset_index} not found")
        
def plot_yolo_keypoints(dataset_dir, split='train', num_images=5, plot_visible_only=True, visibility_threshold=0.5):
    """
    Plots a specified number of images with their YOLOv8 keypoint annotations.
    
    Args:
        dataset_dir (str): Path to the directory containing the dataset.
        split (str): Dataset split to use ('train', 'val', 'test').
        num_images (int): Number of images to plot.
        plot_visible_only (bool): If True, plots only visible keypoints; otherwise, plots all keypoints.
        visibility_threshold (float): Threshold for considering a keypoint visible (0.0 to 1.0).
    """
    
    # Construct paths for images and labels based on the specified split
    image_dir = os.path.join(dataset_dir, split, 'images') if os.path.isdir(os.path.join(dataset_dir, split, 'images')) else os.path.join(dataset_dir, 'images', split)
    label_dir = os.path.join(dataset_dir, split, 'labels') if os.path.isdir(os.path.join(dataset_dir, split, 'labels')) else os.path.join(dataset_dir,'labels', split)

    # Get the list of image and label files
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.webp', '.jpeg', '.bmp'))])
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])

    # Randomly select a subset of images
    selected_indices = random.sample(range(len(image_files)), min(num_images, len(image_files)))

    # Plot the selected images with their keypoints
    for idx in selected_indices:
        image_path = os.path.join(image_dir, image_files[idx])
        label_path = os.path.join(label_dir, label_files[idx])

        # Load the image
        image = Image.open(image_path)
        img_width, img_height = image.size

        # Load the labels
        with open(label_path, 'r') as f:
            labels = f.readlines()

        # Create a new figure
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')

        # Plot keypoints for each object in the image
        for label in labels:
            data = list(map(float, label.strip().split()))
            class_id, x_center, y_center, width, height = data[:5]
            keypoints = data[5:]

            # Calculate bounding box coordinates
            x1 = int((x_center - width/2) * img_width)
            y1 = int((y_center - height/2) * img_height)
            x2 = int((x_center + width/2) * img_width)
            y2 = int((y_center + height/2) * img_height)

            # Draw bounding box
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='r', linewidth=2))

            # Plot keypoints
            for i in range(0, len(keypoints), 3):
                x, y, visibility = keypoints[i:i+3]
                x = int(x * img_width)
                y = int(y * img_height)

                if plot_visible_only and visibility < visibility_threshold:
                    continue

                color = 'g' if visibility >= visibility_threshold else 'y'
                plt.plot(x, y, 'o', color=color, markersize=5)

        plt.title(f'Image: {image_files[idx]}')
        plt.show()

def analyze_yolo_dataset(dataset_dir):
    """
    Analyzes the number of images in each split of a YOLOv8 dataset,
    creates a DataFrame with counts and percentages, and plots a pie chart.

    Args:
        dataset_dir (str): Path to the directory containing the dataset.

    Returns:
        pandas.DataFrame: A DataFrame containing the analysis results.
    """
    if not os.path.exists(dataset_dir):
        return "Invalid dataset path."
    yolo_dataset_structure = 0 if os.path.isdir(os.path.join(dataset_dir, "images")) else 1

    # Define the splits to analyze
    splits = [name for name in os.listdir(dataset_dir)] if yolo_dataset_structure==1 else [name for name in os.listdir(os.path.join(dataset_dir, "images"))]
    
    if 'train' not in splits:
        return "Invalid dataset structure."

    # Initialize a dictionary to store the image counts
    image_counts = {}

    # Count images in each split
    for split in splits:
        split_dir = os.path.join(dataset_dir, split, 'images') if yolo_dataset_structure ==1 else os.path.join(dataset_dir, 'images', split)
        if os.path.exists(split_dir):
            image_files = [f for f in os.listdir(split_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            image_counts[split] = len(image_files)
        else:
            image_counts[split] = 0

    # Calculate total images and percentages
    total_images = sum(image_counts.values())
    percentages = {
                        split: (count / total_images * 100 if total_images > 0 else 0)
                        for split, count in image_counts.items()
                    }

    # Create a DataFrame
    df = pd.DataFrame({
        'Split': splits,
        'Image Count': [image_counts[split] for split in splits],
        'Percentage': [percentages[split] for split in splits]
    })

    # Sort the DataFrame by Image Count in descending order
    df = df.sort_values('Image Count', ascending=False).reset_index(drop=True)

    # Format the Percentage column
    df['Percentage'] = df['Percentage'].apply(lambda x: f"{x:.2f}%")

    # Create a pie chart
    plt.figure(figsize=(3, 2))
    plt.pie(df['Image Count'], labels=df['Split'], autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of Images Across Splits')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

    # Display the plot
    plt.show()

    return df

# Plots images within a path to a 4xn grid using Matplotlib
def plot_4x(directory):
    files = os.listdir(directory)
    png_files = [file for file in files if file.lower().endswith('.png') or file.lower().endswith('.jpg')]

    num_images = len(png_files)
    cols = 4
    rows = (num_images + cols - 1) // cols

    dpi = 400

    fig, axes = plt.subplots(rows, cols, figsize=(12, 8), dpi=dpi)

    for i, ax in enumerate(axes.flat):
        if i < num_images:
            image_path = os.path.join(directory, png_files[i])
            ax.imshow(plt.imread(image_path))
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()
    
def yolo_to_coco(yolo_dir):
    """
    Converts YOLO annotations to COCO format for keypoint or object detection.
    Args:
        yolo_dir (str): Path to the directory containing the YOLO annotations.
        out_dir (str): Path to the output directory where the JSON annotations will be stored.
    """
    yolo_dataset_structure = 0 if os.path.exists(os.path.join(yolo_dir, "images")) else 1
    # Read class names from config file
    config_path = os.path.join(yolo_dir, "data.yaml") 
    with open(config_path, 'r') as f:
        class_data = yaml.safe_load(f)
    if isinstance(class_data['names'], dict):
        class_names = [class_data['names'][key] for key in sorted(class_data['names'].keys())]
    else:
        class_names = class_data['names']
        
    def process_phase(phase):
        images = []
        annotations = []
        img_id = 0
        annotation_id = 0
        imgs_path = os.path.join(yolo_dir, "images", phase) if yolo_dataset_structure==0 else os.path.join(yolo_dir, phase, "images")
        img_list = os.listdir(imgs_path)
        for img_name in img_list:
            img_path = os.path.join(imgs_path, img_name)
            try:
                with Image.open(img_path) as img:
                    img_width, img_height = img.size
            except IOError:
                print(f"Error opening image {img_path}. Skipping.")
                continue

            images.append({
                "id": img_id,
                "width": img_width,
                "height": img_height,
                "file_name": os.path.join("images", phase, img_name)
            })

            label_path = os.path.join(imgs_path.replace("images", "labels"), os.path.splitext(img_name)[0] + ".txt")
            
            try:
                label_data = np.loadtxt(fname=label_path, delimiter=" ", ndmin=2)
            except IOError:
                print(f"Error reading label file {label_path}. Skipping.")
                continue

            keypoints = []
            cls_id = []
            
            if label_data.size > 0:
                parts = label_data[0]
                num_keypoints = (len(parts) - 5) // 3  # Number of keypoints
                class_id = float(parts[0])
                x_center = float(parts[1]) * img_width  # Denormalizing x_center
                y_center = float(parts[2]) * img_height # Denormalizing y_center
                width = float(parts[3]) * img_width     # Denormalizing width
                height = float(parts[4]) * img_height   # Denormalizing height
                x_min = int(x_center - width / 2)
                y_min = int(y_center - height / 2)
                # x_max = x_center + width / 2
                # y_max = y_center + height / 2
                
                cls_id.append([class_id,])
                # bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
                visible_kpts = 0
                for i in range(num_keypoints):
                    x = float(parts[5 + 3*i]) * img_width    # Denormalizing x
                    y = float(parts[6 + 3*i]) * img_height   # Denormalizing y
                    visibility = int(parts[7 + 3*i])
                    keypoints.extend([int(x), int(y), visibility])
                    if visibility>0:
                        visible_kpts+=1
            else:
                print(f"Empty label file {label_path}. Skipping.")
                continue
            if num_keypoints > 0:
                annotations.append({
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": int(class_id),
                    "bbox": [int(x_min), int(y_min), int(width), int(height)],
                    "area": width * height,
                    "iscrowd": 0,
                    "keypoints": keypoints,
                    "num_keypoints": visible_kpts
                })
                annotation_id += 1
                img_id += 1
                categories = [{"id": i, "name": name, "supercategory": "none", "keypoints": [f"kp_{i+1}" for i in range(num_keypoints)], "skeleton": []} for i, name in enumerate(class_names)]
                
            else:
                annotations.append({
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": int(class_id),
                    "bbox": [x_min, y_min, width, height],
                    "area": width * height,
                    "iscrowd": 0
                })
                annotation_id += 1
                img_id += 1
                categories = [{"id": i, "name": name} for i, name in enumerate(class_names)]
        return images, annotations, categories

    # Process train and val sets separately
    split_dir = os.path.join(yolo_dir, "images") if yolo_dataset_structure==0 else yolo_dir
    dirnames = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
    for dirname in dirnames:
        images, annotations, categories = process_phase(dirname)
        # Save annotations
        save_path = os.path.join(yolo_dir, dirname, "labels", f"annotations.json") if yolo_dataset_structure==1 else os.path.join(yolo_dir,"labels", dirname, f"annotations.json")
        with open(save_path, 'w') as outfile:
            json.dump({
                "images": images,
                "annotations": annotations,
                "categories": categories
            }, outfile, indent=4)
            

def extract_frames(video_path, resize_size=None):
    """
    Extracts frames from a video file and optionally resizes them.
    Parameters:
    video_path (str): Path to the input video file.
    resize_size (int, optional): If provided, resizes the largest dimension of the frames to this size while maintaining aspect ratio.
    Returns:
    tuple: A tuple containing:
        - save_dir (str): Directory where the extracted frames are saved.
        - fps (float): Frames per second of the input video.
    Raises:
    FileNotFoundError: If the video file does not exist.
    Notes:
    - The function uses `ffprobe` to retrieve the frame rate of the video.
    - Frames are extracted using `ffmpeg` and saved as JPEG images in a directory named "extracted" located in the same directory as the input video.
    - If `resize_size` is provided, the frames are resized such that the largest dimension is equal to `resize_size` while maintaining the aspect ratio.
    """

    if not os.path.exists(video_path):
        print("Video Not Found")
        return None, None
    # 1) Retrieve FPS using ffprobe
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "csv=p=0",
        video_path
    ]
    result = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    r_frame_rate = result.stdout.strip()  # e.g. "30/1"
    
    if "/" in r_frame_rate:
        num, den = r_frame_rate.split("/")
        fps = float(num) / float(den) if float(den) != 0 else float(num)
    else:
        fps = float(r_frame_rate)

    # 2) Build ffmpeg command for frame extraction
    save_dir = os.path.join(os.path.dirname(video_path), "extracted")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    output_pattern = os.path.join(save_dir, "%d.jpg")

    ffmpeg_cmd = [
        "ffmpeg", "-i", video_path,
        "-vsync", "0",    # Extract every frame without duplication/dropping
    ]

    # 3) Apply scale if resize_size is provided
    if resize_size:
        # This expression scales the largest dimension to `resize_size` while keeping aspect ratio.
        # If width > height, width becomes `resize_size` and height is set to -1.
        # Otherwise, height is `resize_size` and width is -1.
        scale_filter = (
            f"scale='if(gt(iw,ih),{resize_size},-1)':"
            f"'if(gt(iw,ih),-1,{resize_size})'"
        )
        ffmpeg_cmd += ["-vf", scale_filter]

    ffmpeg_cmd += [output_pattern]

    # 4) Run ffmpeg to perform the extraction
    subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return save_dir, fps

def line_angle(line):
    x1, y1, x2, y2 = line
    return np.arctan2(y2 - y1, x2 - x1)

def appx_best_fit_ngon(mask_cv2, n: int = 4) -> list[(int, int)]:
    # convex hull of the input mask
    # mask_cv2_gray = cv2.cvtColor(mask_cv2, cv2.COLOR_RGB2GRAY)
    contours, _ = cv2.findContours(
        mask_cv2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    hull = cv2.convexHull(contours[0])
    hull = np.array(hull).reshape((len(hull), 2))

    # to sympy land
    hull = [sympy.Point(*pt) for pt in hull]

    # run until we cut down to n vertices
    while len(hull) > n:
        best_candidate = None

        # for all edges in hull ( <edge_idx_1>, <edge_idx_2> ) ->
        for edge_idx_1 in range(len(hull)):
            edge_idx_2 = (edge_idx_1 + 1) % len(hull)

            adj_idx_1 = (edge_idx_1 - 1) % len(hull)
            adj_idx_2 = (edge_idx_1 + 2) % len(hull)

            edge_pt_1 = sympy.Point(*hull[edge_idx_1])
            edge_pt_2 = sympy.Point(*hull[edge_idx_2])
            adj_pt_1 = sympy.Point(*hull[adj_idx_1])
            adj_pt_2 = sympy.Point(*hull[adj_idx_2])

            subpoly = sympy.Polygon(adj_pt_1, edge_pt_1, edge_pt_2, adj_pt_2)
            angle1 = subpoly.angles[edge_pt_1]
            angle2 = subpoly.angles[edge_pt_2]

            # we need to first make sure that the sum of the interior angles the edge
            # makes with the two adjacent edges is more than 180°
            if sympy.N(angle1 + angle2) <= sympy.pi:
                continue

            # find the new vertex if we delete this edge
            adj_edge_1 = sympy.Line(adj_pt_1, edge_pt_1)
            adj_edge_2 = sympy.Line(edge_pt_2, adj_pt_2)
            intersect = adj_edge_1.intersection(adj_edge_2)[0]

            # the area of the triangle we'll be adding
            area = sympy.N(sympy.Triangle(edge_pt_1, intersect, edge_pt_2).area)
            # should be the lowest
            if best_candidate and best_candidate[1] < area:
                continue

            # delete the edge and add the intersection of adjacent edges to the hull
            better_hull = list(hull)
            better_hull[edge_idx_1] = intersect
            del better_hull[edge_idx_2]
            best_candidate = (better_hull, area)

        if not best_candidate:
            raise ValueError("Could not find the best fit n-gon!")

        hull = best_candidate[0]

    # back to python land
    hull = [(int(x), int(y)) for x, y in hull]

    return hull

def is_dominant_white(image, bbox):
    """
    Determines if the dominant color within the given bounding box is white.
    
    Parameters:
    - image: numpy.ndarray, input image in BGR format.
    - bbox: tuple, bounding box coordinates as (x_min, y_min, x_max, y_max).
    
    Returns:
    - bool: True if white is the dominant color, False otherwise.
    """
    # Convert PIL image to cv2 image
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    x_min, y_min, x_max, y_max = bbox
    # Extract the region of interest from the image
    roi = image[y_min:y_max, x_min:x_max]
    
    # Check if the ROI is empty (invalid bounding box)
    if roi.size == 0:
        return False
    
    # Convert the ROI to HSV color space
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Split into individual channels
    h, s, v = cv2.split(hsv_roi)
    
    # Define thresholds for saturation and value to identify white pixels
    saturation_threshold = 80
    value_threshold = 100
    dominance_threshold = 0.5  # 70% of the ROI must be white-like
    
    # Create masks for low saturation and high value
    low_saturation = s < saturation_threshold
    high_value = v > value_threshold
    white_pixels = low_saturation & high_value
    
    # Calculate the proportion of white pixels
    white_ratio = np.sum(white_pixels) / white_pixels.size
    
    is_ball = white_ratio >= dominance_threshold
    
    return is_ball

def segment_dominant_color(
    image,
    hue_padding=10,
    sat_padding=80, 
    val_padding=40,
    ignore_low_sat=25,
    ignore_low_val=25
):
    """
    Segments the table in 'image_path' by:
      1. Building a histogram of HSV pixels.
      2. Finding the histogram peak as the 'dominant' color.
      3. Creating a tolerance range around that color to mask the table.
    
    Parameters
    ----------
    image_path: str
        Path to the input image.
    hue_padding: int
        Range around the dominant hue (in [0..179]) to include in the mask.
    sat_padding: int
        Range around the dominant saturation (in [0..255]) to include in the mask.
    val_padding: int
        Range around the dominant value (in [0..255]) to include in the mask.
    ignore_low_sat: int
        Ignore pixels with saturation below this threshold (to skip near-gray).
    ignore_low_val: int
        Ignore pixels with value below this threshold (to skip near-black).
    
    Returns
    -------
    segmented_image: np.ndarray
        BGR image with only the dominant table color region visible.
    mask: np.ndarray
        Binary mask identifying the table region.
    """
    
    # Read image and convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Flatten the HSV image into a list of pixels:
    # shape is (height*width, 3)
    hsv_flat = hsv.reshape(-1, 3)

    # Optionally filter out near-gray or near-black pixels to avoid muddying the histogram
    # for example, ignoring background or shadows.
    mask_filter = (hsv_flat[:, 1] >= ignore_low_sat) & (hsv_flat[:, 2] >= ignore_low_val)
    hsv_filtered = hsv_flat[mask_filter]

    # If everything got filtered out, fallback to a raw histogram
    if len(hsv_filtered) == 0:
        hsv_filtered = hsv_flat

    # Build a 3D histogram in H, S, V
    # Ranges:
    #   Hue: [0..180)
    #   Sat: [0..256)
    #   Val: [0..256)
    # We can keep a coarser bin size to reduce noise, e.g. 180 for hue, 256 for sat/val
    hist_3d, edges = np.histogramdd(
        hsv_filtered.astype(np.float32), 
        bins=(180, 256, 256), 
        range=((0, 180), (0, 256), (0, 256))
    )

    # Find the bin with the largest count. This is the 'mode' (peak) of the color distribution.
    # The index returned will be (h_idx, s_idx, v_idx).
    peak_idx = np.unravel_index(np.argmax(hist_3d), hist_3d.shape)

    # Convert bin index to approximate color value
    #   each bin has a certain width in each dimension:
    #   - hue_width ~ 1.0
    #   - sat_width ~ 1.0
    #   - val_width ~ 1.0
    # For integer bins, the center of bin i is roughly i.
    # Adjust as you see fit; here we treat the bin index as the actual HSV.
    dominant_h = peak_idx[0]
    dominant_s = peak_idx[1]
    dominant_v = peak_idx[2]

    # Build a tolerance range around the dominant color
    lower_h = max(dominant_h - hue_padding, 0)
    upper_h = min(dominant_h + hue_padding, 179)
    lower_s = max(dominant_s - sat_padding, 0)
    upper_s = min(dominant_s + sat_padding, 255)
    lower_v = max(dominant_v - val_padding, 0)
    upper_v = min(dominant_v + val_padding, 255)

    lower_bound = np.array([lower_h, lower_s, lower_v], dtype=np.uint8)
    upper_bound = np.array([upper_h, upper_s, upper_v], dtype=np.uint8)

    # Create mask using the dynamic range
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # (Optional) Refine with morphological operations to clean up noise
    # kernel = np.ones((5, 5), np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Apply mask
    segmented_image = cv2.bitwise_and(image, image, mask=mask)

    return segmented_image

def corner_distances(corners):
    """
    corners: ndarray (4, 2) => [[x0, y0],
                               [x1, y1],
                               [x2, y2],
                               [x3, y3]]
    returns: list of 6 floats (all unique distances between the 4 points)
    """
    pairs = [(0,1), (1,2), (2,3), (3,0), (0,2), (1,3)]
    dists = []
    for (i, j) in pairs:
        d = np.linalg.norm(corners[i] - corners[j])
        dists.append(d)
    return dists

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def numeric_sort(value):
    numbers = re.findall(r'\d+', value)
    return int(numbers[0]) if numbers else -1