import gc
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

from utils import extract_frames, is_dominant_white, show_mask

sam_model_path="../sam2/checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
class BallTracker:
    def __init__(self, video_path, save_dir, visualize=False):
        self.save_dir = save_dir
        self.video_path = video_path
        self.frame_paths = sorted([os.path.join(self.save_dir, p) for p in os.listdir(save_dir) if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]], key=lambda x: int(os.path.splitext(os.path.basename(x))[0])) # scan all the JPEG frame names in this directory
        self.sam_predictor = build_sam2_video_predictor(model_cfg, sam_model_path, device=device, offload_video_to_cpu=True, async_loading_frames=True)
        self.visualize = visualize
            
    def find_ball_frame(self):
        try:
            image_processor = RTDetrImageProcessor.from_pretrained("jadechoghari/RT-DETRv2", cache_dir="../chkpts")
            model = RTDetrForObjectDetection.from_pretrained("jadechoghari/RT-DETRv2", cache_dir="../chkpts")
            cap = cv2.VideoCapture(self.video_path)
            frame_idx = 0
            if self.frame_paths:
                sample_image = cv2.imread(self.frame_paths[0])
                scaled_height, scaled_width = sample_image.shape[:2]
            else:
                raise ValueError("No frames found in save_dir")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                inputs = image_processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                results = image_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([image.size[::-1]]), threshold=0.5)

                for result in results:
                    for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
                        if label_id.item() == 32:
                            box = [int(i) for i in box.tolist()]
                            if is_dominant_white(frame, box):
                                print(f"Found the ball at frame {frame_idx} with box: {box}")
                                if self.visualize:
                                    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                                    plt.gca().add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], edgecolor='green', facecolor='none', linewidth=2))
                                    plt.show()
                                    
                                # Rescale the box to the resized frame dimensions (floats, not integers)
                                box = [
                                    max(0.0, min(box[0] * scaled_width / image.size[0], scaled_width - 1)),
                                    max(0.0, min(box[1] * scaled_height / image.size[1], scaled_height - 1)),
                                    max(0.0, min(box[2] * scaled_width / image.size[0], scaled_width - 1)),
                                    max(0.0, min(box[3] * scaled_height / image.size[1], scaled_height - 1)),
                                ]
                                final_box = np.array([np.float32(int(p)) for p in box], dtype=np.float32)
                                print(f"Final box: {final_box}")
                                return frame_idx, final_box
                frame_idx += 1
            return None, None
            
        finally:
            cap.release()
            del model
            del image_processor
            torch.clear_autocast_cache()
            torch.cuda.empty_cache()
            gc.collect()
        
    def track_ball(self, frame_idx, bbox):
        # frame_idx, bbox = self.find_ball_frame()
        try:
            if frame_idx is None or bbox is None:
                print("Ball can't be found")
                return
            
            # run propagation throughout the video and collect the results in a dict
            inference_state = self.sam_predictor.init_state(video_path=self.save_dir)
            ann_obj_id = 1
            self.sam_predictor.add_new_points_or_box(
                                inference_state=inference_state,
                                frame_idx=frame_idx,
                                obj_id=ann_obj_id,
                                box=bbox,
                            )
            video_segments = {}  # video_segments contains the per-frame segmentation results
            for out_frame_idx, out_obj_ids, out_mask_logits in self.sam_predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            if self.visualize:
                # render the segmentation results every few frames
                vis_frame_stride = 60
                # plt.close("all")
                for out_frame_idx in range(frame_idx, len(self.frame_paths), vis_frame_stride):
                    frame = cv2.imread(self.frame_paths[out_frame_idx])
                    plt.figure(figsize=(6, 4))
                    plt.title(f"frame {out_frame_idx}")
                    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
                        
            ball_detections = {}
            for seg_frame_idx, seg in video_segments.items():
                # Ensure a list is initialized for this frame
                ball_detections[seg_frame_idx] = []
                for obj_id, mask in seg.items():
                    y, x = np.where(mask[0])
                    if len(x) > 0 and len(y) > 0:
                        center_x = int(np.mean(x))
                        center_y = int(np.mean(y))
                        ball_detections[seg_frame_idx].append((center_x, center_y, 1))
                    else:
                        ball_detections[seg_frame_idx].append((np.nan, np.nan, -1))

            data = []
            for frame, detections in ball_detections.items():
                if detections:  # If there is at least one detection
                    # Here, we select the first detection.
                    x, y, status = detections[0]
                else:
                    x, y, status = np.nan, np.nan, -1
                data.append({'frame': frame, 'x': x, 'y': y, 'status': status})

            # Create the DataFrame and set the frame as index.
            df = pd.DataFrame(data).set_index('frame')

            # Convert x and y to numeric types (this is crucial for interpolation).
            df['x'] = pd.to_numeric(df['x'], errors='coerce')
            df['y'] = pd.to_numeric(df['y'], errors='coerce')

            # Interpolate missing values in x and y.
            df[['x', 'y']] = df[['x', 'y']].interpolate(method='linear', limit_direction='both')

            # Convert back to the original format (tuple for each frame).
            # Here, we reassemble each row into a tuple (x, y, 1) if values exist.
            interpolated_ball_detections = df.apply(
                lambda row: (row['x'], row['y'], 1) if not pd.isna(row['x']) and not pd.isna(row['y'])
                else (np.nan, np.nan, -1),
                axis=1
            ).to_dict()

            return interpolated_ball_detections

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            if 'inference_state' in locals():
                self.sam_predictor.reset_state(inference_state)
            del self.sam_predictor
            torch.clear_autocast_cache()
            torch.cuda.empty_cache()
if __name__ == "__main__":
    save_dir, _ = extract_frames("../media/clip2.mp4", resize_size=240)
    bt = BallTracker(video_path="../media/clip2.mp4", save_dir=save_dir, visualize=True)
    frame_idx, bbox = bt.find_ball_frame()
    ball_detections = bt.track_ball(frame_idx, bbox)
