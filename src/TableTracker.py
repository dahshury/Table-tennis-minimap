import gc
import os

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from ultralytics import YOLO
from utils import appx_best_fit_ngon, extract_frames, numeric_sort, segment_dominant_color

class TableTracker:
    """ A class to track a table in a video using YOLO for object detection and SAM for segmentation.
        Attributes:
            video_path (str): Path to the input video file.
            yolo_model_path (str): Path to the YOLO model weights.
            sam_model_predictor (SAM2ImagePredictor): SAM model predictor for image segmentation.
            visualize (bool): Flag to enable visualization of intermediate results.
            fps (int): Frames per second of the input video.
            frames (list): List of frames extracted from the video.
        Methods:
            __init__(self, video_path, yolo_model_path, sam_model_path, sam_config_path, visualize):
                Initializes the TableTracker with the given parameters.
            detect_table_bbox(self, image):
                Detects the bounding box of the table in the given image using YOLO.
            segment_table(self, image, input_point=None, input_label=None, input_box=None):
                Segments the table in the given image using SAM.
            detect_table_edges(self, idx, num_secs=7):
                Detects the edges of the table in a span of frames around the given index.
            track_table_points(self, table_pts, distance_threshold=100):
                Tracks the table points through the entire video using feature-based RANSAC homography.
            process_video(self, output_video=False):
                Main pipeline to process the video, detect table corners, track them, and optionally write an output video.
            visualize_results(self, image, mask_crop_image, denoised_image, segmented, closed, hull_image):
                Visualizes the intermediate results of the table detection and segmentation process."""
    
    def __init__(self, video_path, yolo_model_path="../runs/pose/train5/weights/last.pt",
                 sam_model_path="../sam2/checkpoints/sam2.1_hiera_large.pt",
                 sam_config_path="configs/sam2.1/sam2.1_hiera_l.yaml",
                 visualize=False):
        self.video_path = video_path
        self.table_yolo_model =  YOLO(yolo_model_path)
        self.sam_model_predictor = SAM2ImagePredictor(build_sam2(sam_config_path, sam_model_path))
        self.visualize= visualize
        save_dir, self.fps = extract_frames(self.video_path, resize_size=None)
        self.frames = [
            cv2.imread(os.path.join(save_dir, f))
            for f in sorted(os.listdir(save_dir), key=numeric_sort)
        ]
    def detect_table_bbox(self, image):
        
        results = self.table_yolo_model.predict(source=image)
        bbox_table = None
        for result in results:
            for box in result.boxes:
                # Just take the first bounding box found
                bbox_table = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                break
        return bbox_table

    def segment_table(self, image, input_point=None, input_label=None, input_box=None):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.sam_model_predictor.set_image(image)
            masks, scores, logits = self.sam_model_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=input_box,
                multimask_output=True,
            )
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            scores = scores[sorted_ind]
            logits = logits[sorted_ind]
        return masks, scores, logits

    def detect_table_edges(self, idx, num_secs=7):
        """
        1. Search a span of frames around 'idx' to find the best table mask (largest area).
        2. Return the corner points for the best mask, sorted such that:
        - The points on the left (relative to a normalized direction_line) come first,
        - Ordered as: top-left, bottom-left, bottom-right, top-right.
        
        (If a direction_line is available, it is rotated so that it points between 0 and 90° 
        relative to the positive x-axis.)
        """
        biggest_mask_area = 0
        best_mask = None
        best_image = None
        best_idx = -1

        # ------------------------------------------------------------------
        # 1. Find "best" frame for table detection in next num_secs of video
        # ------------------------------------------------------------------
        for local_i, frame in enumerate(
            self.frames[idx: int(idx + self.fps * num_secs)
                        if idx + self.fps * num_secs < len(self.frames)
                        else len(self.frames)]
        ):
            # Skip some frames to speed up searching
            if local_i % 10 != 0:
                continue

            table_bbox = self.detect_table_bbox(frame)
            if not table_bbox:
                continue

            table_masks, table_scores, table_logits = self.segment_table(
                image=frame,
                input_label=np.array([1]),
                input_box=table_bbox
            )
            highest_score_mask_index = np.argmax(table_scores)
            highest_socre_mask = table_masks[highest_score_mask_index]
            mask_area = np.sum(highest_socre_mask)
            if mask_area > biggest_mask_area:
                biggest_mask_area = mask_area
                best_mask = highest_socre_mask
                best_image = frame
                best_idx = idx + local_i

        if best_idx == -1 or best_mask is None:
            print("No suitable table mask found.")
            return None

        # -------------------------------
        # 2. Post-process to get corners
        # -------------------------------
        # Apply the mask to get the cropped image
        mask_crop_image = cv2.bitwise_and(best_image, best_image, mask=best_mask.astype(np.uint8))

        # Denoise and segment
        denoised_image = cv2.fastNlMeansDenoising(mask_crop_image, h=40)
        segmented = segment_dominant_color(denoised_image)
        segmented_bw = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)

        # Thresholding and morphological operations
        thresholded = cv2.threshold(segmented_bw, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        struc1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        mask_crop_gray = cv2.cvtColor(mask_crop_image, cv2.COLOR_BGR2GRAY)
        opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, struc1)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, struc1, iterations=1)
        edges = cv2.Canny(closed, 50, 100)
        # edges_2 = cv2.Canny(mask_crop_gray, 50, 100)

        contours, _ = cv2.findContours(edges.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            print("No contours found.")
            return None

        all_points = np.vstack(contours)
        hull = cv2.convexHull(all_points)
        hull_image = np.zeros_like(mask_crop_image)
        cv2.drawContours(hull_image, [hull], -1, (0, 255, 0), 2)
        poly = appx_best_fit_ngon(cv2.cvtColor(hull_image, cv2.COLOR_BGR2GRAY), 4)
        
        # We'll store corners in a dict: {best_idx: [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]}
        table_pts = {
            best_idx: [(int(pt[0]), int(pt[1])) for pt in poly]
        }
        

        # Optional visualization
        if self.visualize:
            self.visualize_results(
                image=self.frames[best_idx],
                mask_crop_image=mask_crop_image,
                denoised_image=denoised_image,
                segmented=segmented,
                closed=closed,
                hull_image=hull_image,
            )
        return table_pts

    def track_table_points(self, table_pts, distance_threshold=100):
        """
        table_pts: dict with a single key = reference frame index, 
                   value = list of 4 corners (x, y).
                   
        We'll do a feature-based approach using ORB + RANSAC homography:
          1. Extract features in the reference frame.
          2. For each other frame, detect features, match them, compute homography H with RANSAC.
          3. Warp the reference corners by H.
          4. Save the resulting corners in a dict { frame_idx: corners }.
        """
        # --------------------------------
        # 1. Prepare reference data
        # --------------------------------

        # ORB initialization
        orb = cv2.ORB_create(nfeatures=2000)

        # We'll store corners for each frame
        tracked_table_points = table_pts.copy()

        # --------------------------------
        # 2. Function to compute corners via RANSAC for each new frame
        # --------------------------------
        def compute_homography_and_transform_corners(ref_frame_idx, curr_frame_idx):
            """
            Return the new corners for the corners in ref_corners by applying 
            the RANSAC homography from reference_frame -> current_frame.
            """
            
            ref_frame = self.frames[ref_frame_idx]
            curr_frame = self.frames[curr_frame_idx]
            
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
            ref_corners = np.array(tracked_table_points[ref_frame_idx], dtype=np.float32).reshape(-1, 1, 2)
            
            # Detect and compute with ORB
            keypoints_ref, descriptors_ref = orb.detectAndCompute(ref_gray, None)
            keypoints_curr, descriptors_curr = orb.detectAndCompute(curr_gray, None)
            if descriptors_curr is None or len(keypoints_curr) < 4:
                return None  # not enough features

            # BFMatcher to match features
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(descriptors_ref, descriptors_curr)
            # ratio = 0.7  # typical ratio test value
            if not matches:
                return None

            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)[:100]
            # You might want to filter top-k, e.g., top 200, or use ratio test etc.

            # Coordinates in reference frame
            ref_pts = np.float32([keypoints_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            # Coordinates in current frame
            curr_pts = np.float32([keypoints_curr[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Find homography with RANSAC
            H, mask = cv2.findHomography(ref_pts, curr_pts, cv2.RANSAC, 1)
            if H is None:
                return None

            # Warp the reference corners
            new_corners = cv2.perspectiveTransform(ref_corners, H)
            # Convert corners to a Python list of (x,y)
            new_corners_array = np.array([
                [int(pt[0][0]), int(pt[0][1])] for pt in new_corners
            ])
            return new_corners_array

        # --------------------------------
        # 3. Track for all frames
        # --------------------------------
        # A. Backward propagation (ref_idx -> 0)
        init_ref_idx = next(iter(tracked_table_points.keys()))
        ref_corners = np.array(tracked_table_points[init_ref_idx], dtype=np.float32).reshape(-1, 1, 2)
        ref_index = init_ref_idx
        for i in range(init_ref_idx - 1, -1, -1):
            new_corners = compute_homography_and_transform_corners(ref_index, i)
            if new_corners is None:
                print("Falling back to previous frame.")
                # fallback: re-use corners from previous known frame
                tracked_table_points[i] = tracked_table_points[i + 1]
                continue
            new_corners_2d = np.array(new_corners, dtype=np.float32).reshape(-1, 2)
            tracked_table_points[i] = new_corners
            moved_corners = any(np.linalg.norm(pp - cp) > distance_threshold for pp, cp in zip(ref_corners, new_corners_2d))
            if moved_corners:
                ref_corners = tracked_table_points[i + 1]
                ref_index = i + 1
            
        # B. Forward propagation (ref_idx -> end)
        ref_corners = np.array(tracked_table_points[init_ref_idx], dtype=np.float32).reshape(-1, 1, 2)
        ref_index = init_ref_idx
        for i in range(init_ref_idx + 1, len(self.frames)):
            new_corners = compute_homography_and_transform_corners(ref_index, i)
            if new_corners is None:
                print("Falling back to previous frame.")
                # fallback: re-use corners from previous known frame
                tracked_table_points[i] = tracked_table_points[i - 1]
                continue
            new_corners_2d = np.array(new_corners, dtype=np.float32).reshape(-1, 2)
            tracked_table_points[i] = new_corners
            moved_corners = any(np.linalg.norm(pp - cp) > distance_threshold for pp, cp in zip(ref_corners, new_corners_2d))
            if moved_corners:
                ref_corners = tracked_table_points[i - 1]
                ref_index = i - 1
        return tracked_table_points

    def process_video(self, output_video=False):
        """
        Main pipeline:
          1. Loop frames to find the table bounding box & corners in one “best” frame.
          2. Use that frame’s corners + feature-based RANSAC homography to track corners through entire video.
          3. Write an output video with the tracked corners drawn on each frame.
        """
        try:
            # 1. Detect table corners in some frame
            table_pts = None
            tracked_points = None
            for idx, frame in enumerate(self.frames):
                bbox = self.detect_table_bbox(frame)
                if bbox:
                    table_pts = self.detect_table_edges(idx)
                    if table_pts:
                        print(f"Found table corners at frame {next(iter(table_pts.keys()))}")
                        tracked_points = self.track_table_points(table_pts)
                        if not output_video:
                            return tracked_points
                        else:
                            break
                    else:
                        print("Table not found.")
                        return None
                            
            if output_video and tracked_points:
                output_path = os.path.join(os.path.dirname(self.video_path), "output.avi")
                output_dir = os.path.dirname(output_path)
                if not os.path.exists(output_dir):
                    print(f"Creating output directory: {output_dir}")
                    os.makedirs(output_dir, exist_ok=True)

                if not os.access(output_dir, os.W_OK):
                    print(f"NO WRITE PERMISSIONS for: {output_dir}")
                    return
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                out = cv2.VideoWriter(
                    output_path,
                    fourcc,
                    self.fps,
                    (int(self.frames[0].shape[1]), int(self.frames[0].shape[0]))
                )
                # Check if VideoWriter is opened successfully
                if not out.isOpened():
                    print("Error: Failed to open VideoWriter. Check codec support or output path.")
                    return

                # 3. Write output video frames, drawing the tracked corners
                for i, frame in enumerate(self.frames):
                    corners_i = tracked_points.get(i, None)
                    if corners_i is not None:
                        for pt in corners_i:
                            x, y = int(pt[0]), int(pt[1])
                            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                    out.write(frame)

                out.release()
                print(f"Done. Output saved to: {output_path}")
            return tracked_points
        finally:
            # Clean up models and resources
            del self.sam_model_predictor
            del self.table_yolo_model
            torch.cuda.empty_cache()
            gc.collect()
    def visualize_results(self, image, mask_crop_image, denoised_image, segmented, closed, hull_image):
        # Display the results
        plt.figure(figsize=(10, 10))
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        plt.figure(figsize=(10, 10))
        plt.title("SAM Mask")
        plt.imshow(mask_crop_image)
        
        plt.figure(figsize=(10, 10))
        plt.title("Denoised")
        plt.imshow(denoised_image)
        
        plt.figure(figsize=(10, 10))
        plt.title("Segmented")
        plt.imshow(segmented)
        
        plt.figure(figsize=(10, 10))
        plt.title("mask1")
        plt.imshow((cv2.cvtColor(closed, cv2.COLOR_GRAY2RGB)))

        plt.figure(figsize=(10, 10))
        plt.title("Polygon")
        plt.imshow(hull_image)
        
        plt.show()
        cv2.waitKey(0)
        cv2.destroyAllWindows()
