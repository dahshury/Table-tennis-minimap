# MiniTable.py
import cv2
import numpy as np
import os
from collections import deque
from BallTracker import BallTracker
from HumanTracker import HumanTracker
from TableTracker import TableTracker
from utils import extract_frames
from ultralytics import YOLO

class MiniTable:
    """
    A class to process a video of a ping pong game and generate a mini table video with ball and human detections.
    Attributes:
        video_path (str): Path to the input video file.
        table_track (dict): Dictionary containing table tracking data for each frame.
        ball_detections (dict): Dictionary containing ball detection data for each frame.
        human_tracks (dict): Dictionary containing human tracking data for each frame.
        resize_from (int): Maximum dimension to resize the video frames for processing.
        mini_table_image (str): Path to the mini table image.
        human0_buffer (deque): Buffer for temporal averaging of human0 detections.
        human1_buffer (deque): Buffer for temporal averaging of human1 detections.
    Methods:
        find_homography_matrix(src_pts):
            Computes the homography matrix to transform points from the original video to the mini table image.
        get_center(box):
            Computes the center of a bounding box.
        process_video():
            Processes the input video and generates the mini table video with ball and human detections.
    """
    def __init__(self, video_path, table_track, ball_detections, human_tracks, resize_from=None):
        self.video_path = video_path
        self.table_track = table_track
        self.ball_detections = ball_detections  # Each value is (x, y, vis)
        self.human_tracks = human_tracks        # Each value is a list of bboxes or None.
        self.mini_table_image = "../media/full_table.png"
        self.resize_from = resize_from
        
        # Buffers for temporal averaging over 10 frames for human detections.
        self.human0_buffer = deque(maxlen=5)
        self.human1_buffer = deque(maxlen=5)

    def find_homography_matrix(self, src_pts):
        dst_points = np.array([
            [2017, 723],
            [1368, 723],
            [1368, 1077],
            [2017, 1077]
        ], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src_pts, dst_points)
        return M

    @staticmethod
    def get_center(box):
        if box:
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            return (cx, cy)
        return None

    def process_video(self):
        # Open the video file and get original dimensions.
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error opening video file.")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Compute dimensions for extracted frames (max dimension = self.resize_from)
        if self.resize_from:
            if width_orig >= height_orig:
                extracted_width = self.resize_from
                extracted_height = int(round(height_orig * (self.resize_from / width_orig)))
            else:
                extracted_height = self.resize_from
                extracted_width = int(round(width_orig * (self.resize_from / height_orig)))
        
            # Scaling factors to convert from extracted frame coordinates to original dimensions.
            scale_x = width_orig / extracted_width
            scale_y = height_orig / extracted_height
        else:
            scale_x, scale_y = 1,1

        # Load the mini table image.
        mini_table_img = cv2.imread(self.mini_table_image)
        if mini_table_img is None:
            print("Error loading mini table image.")
            return
        height, width, _ = mini_table_img.shape

        # Prepare output video.
        output_video_path = "../media/mini_table_video.mp4"
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        missing_detections = []
        if not self.ball_detections:
            missing_detections.append("ball detections")
        if not self.human_tracks:
            missing_detections.append("human tracks")
        if not self.table_track:
            missing_detections.append("table track")
        
        if missing_detections:
            print(f"Missing detections from: {', '.join(missing_detections)}")
            return
        
        frame_idx = 0
        while True:
            ret, _ = cap.read()  # Frame image not used directly.
            if not ret:
                break

            # Start with a fresh copy of the mini table image.
            frame_mini = mini_table_img.copy()
            
            # Use table_track data (assumed to be in original video coordinate space) to compute the homography.
            if frame_idx in self.table_track:
                src_table_pts = np.array(self.table_track[frame_idx], dtype=np.float32)
                M = self.find_homography_matrix(src_table_pts)
            else:
                M = None
                print(f"Can't find homography matrix for frame {frame_idx}")
                return
            
            # Process ball detection.
            if frame_idx in self.ball_detections:
                ball_data = self.ball_detections[frame_idx]  # (x, y, vis)
                # Rescale ball coordinates from extracted frame space to original dimensions.
                ball_center_xy = (ball_data[0] * scale_x, ball_data[1] * scale_y)
                ball_center_arr = np.array([[ball_center_xy]], dtype=np.float32)
                ball_center_transformed = cv2.perspectiveTransform(ball_center_arr, M)[0][0]
                cv2.circle(frame_mini, (int(ball_center_transformed[0]), int(ball_center_transformed[1])), 
                           10, (85, 44, 255), -1)

            # Update human detection buffers with the current frame's detections.
            if frame_idx in self.human_tracks:
                human_boxes = self.human_tracks[frame_idx]
                # Update buffer for human0.
                if len(human_boxes) > 0 and human_boxes[0] is not None:
                    human0_center = self.get_center(human_boxes[0])
                    if human0_center:
                        self.human0_buffer.append(human0_center)
                # Update buffer for human1.
                if len(human_boxes) > 1 and human_boxes[1] is not None:
                    human1_center = self.get_center(human_boxes[1])
                    if human1_center:
                        self.human1_buffer.append(human1_center)

            # Draw averaged human detections if available.
            if M is not None:
                # Process human0 averaged detection.
                if len(self.human0_buffer) > 0:
                    avg_human0_center = np.mean(np.array(self.human0_buffer), axis=0)
                    avg_human0_center_arr = np.array([[avg_human0_center]], dtype=np.float32)
                    human0_center_transformed = cv2.perspectiveTransform(avg_human0_center_arr, M)[0][0]
                    cv2.circle(frame_mini, (int(human0_center_transformed[0]), int(human0_center_transformed[1])), 
                               20, (14, 10, 84), -1)
                # Process human1 averaged detection.
                if len(self.human1_buffer) > 0:
                    avg_human1_center = np.mean(np.array(self.human1_buffer), axis=0)
                    avg_human1_center_arr = np.array([[avg_human1_center]], dtype=np.float32)
                    human1_center_transformed = cv2.perspectiveTransform(avg_human1_center_arr, M)[0][0]
                    cv2.circle(frame_mini, (int(human1_center_transformed[0]), int(human1_center_transformed[1])), 
                               20, (54, 146, 243), -1)

            # Write the updated mini table image to the output video.
            out.write(frame_mini)
            frame_idx += 1

        cap.release()
        out.release()
        print(f"Mini table video saved to: {output_video_path}")
        
if __name__ == "__main__":
    video_path = "../media/test2.mp4"
    resize_size = None
    
    # Extract frames and process ball tracking
    save_dir, _ = extract_frames(video_path, resize_size=resize_size)
    bt = BallTracker(video_path=video_path, save_dir=save_dir)
    frame_idx, bbox = bt.find_ball_frame()
    ball_track = bt.track_ball(frame_idx, bbox)
    # print(ball_track)
    
    # Process table tracking
    detector = TableTracker(video_path=video_path)
    table_track = detector.process_video(output_video=False)
    # print(table_track)

    # Process human tracking
    model = YOLO("../chkpts/yolo11n.pt")
    human_tracker = HumanTracker(model, video_path, table_track, save_video=False)
    human_tracks = human_tracker.track_players()
    human_tracker.find_most_parallel_line()
    sorted_table_track = human_tracker.sort_points()
    # print(human_tracks)

    # Generate mini table video
    minitable = MiniTable(video_path, sorted_table_track, ball_track, human_tracks, resize_from=resize_size)
    minitable.process_video()