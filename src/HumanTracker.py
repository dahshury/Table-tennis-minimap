import math
from ultralytics import YOLO

class HumanTracker:
    """A class to track human players in a video and analyze their positions relative to a table.
        Attributes:
            model: The tracking model used to process the video.
            video_path: Path to the video file.
            tracked_ids: List of track IDs for the two largest humans.
            save_video: Boolean indicating whether to save the processed video.
            table_track: Dictionary containing table polygon data for each frame.
            human_tracks: Dictionary to store the bounding boxes of the tracked humans for each frame.
            best_edge_idx: Index of the table edge that is most parallel to the line between players.
            best_frame: Frame number where the most parallel line was found.
        Methods:
            track_players():
                Processes the video to track human players and stores their bounding boxes.
            find_most_parallel_line():
                Finds the line between players that is most parallel to any table edge across all frames, used to determine the table's direction.
                Returns a dictionary with the best edge index, frame number, smallest angular difference, 
                points of the best edge, indices of the best edge, and tracked IDs.
            sort_points():
                Re-orders the table_track polygons so that every frame's polygon is sorted relative to the 
                orientation defined by the best edge from the best frame. The desired order is: 
                [top left, bottom left, top right, bottom right].
                Returns a new table_track dictionary with the sorted polygons."""
    
    def __init__(self, model, video_path, table_track, save_video=False):
        self.model = model
        self.video_path = video_path
        self.tracked_ids = []  # Stores the track IDs of the two largest humans
        self.save_video = save_video
        self.table_track = table_track
        self.human_tracks = None
        self.best_edge_idx = None
        self.best_frame = None
    def track_players(self):
        # Process the video with tracking enabled
        results = self.model.track(source=self.video_path, persist=True, save=self.save_video, classes=[0], conf=0.5, tracker="bytetrack.yaml")
        
        # Initialize storage for the two largest humans' tracks
        self.human_tracks = {frame_idx: {0: [], 1: []} for frame_idx in range(len(results))}  # frame_idx: {0: largest, 1: second largest}

        for frame_idx, result in enumerate(results):
            if (
                not result.boxes or 
                result.boxes.xyxy is None or 
                result.boxes.cls is None or 
                result.boxes.id is None
            ):
                if frame_idx == len(results) - 1:
                    return self.human_tracks
                continue  # Skip this frame if any attribute is missing
            # Extract current frame's human detections with track IDs and areas
            humans = []
            for box, cls, track_id in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.id):
                if cls == 0:  # Class 0 for humans
                    x1, y1, x2, y2 = box.tolist()
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    humans.append((track_id, [x1, y1, x2, y2], (cx, cy)))
            
            # Select the two humans closest to the center of the table in the first qualifying frame
            if not self.tracked_ids:
                if len(humans) >= 2:
                    # Calculate the center of the table
                    table_poly = self.table_track.get(frame_idx)
                    if table_poly is None:
                        continue
                    table_center_x = sum([point[0] for point in table_poly]) / 4
                    table_center_y = sum([point[1] for point in table_poly]) / 4

                    # Calculate distances to the table center and sort by distance
                    sorted_humans = sorted(humans, key=lambda x: ((x[2][0] - table_center_x) ** 2 + (x[2][1] - table_center_y) ** 2) ** 0.5)
                    self.tracked_ids = [sorted_humans[0][0], sorted_humans[1][0]]
                    # Record initial frame's bounding boxes
                    self.human_tracks[frame_idx][0] = (sorted_humans[0][1])
                    self.human_tracks[frame_idx][1] = (sorted_humans[1][1])
                    print(f"Selected humans from frame: {frame_idx}")
                else:
                    continue  # Skip until two humans are found
            else:
                # Collect bounding boxes for the tracked IDs in subsequent frames
                id0_box = None
                id1_box = None
                for human in humans:
                    track_id, bbox, _ = human
                    if track_id == self.tracked_ids[0]:
                        id0_box = bbox
                    elif track_id == self.tracked_ids[1]:
                        id1_box = bbox
                
                # Append boxes if found
                if id0_box:
                    self.human_tracks[frame_idx][0] = id0_box
                if id1_box:
                    self.human_tracks[frame_idx][1] = id1_box
        
        return self.human_tracks
    
    def find_most_parallel_line(self):
        """
        Finds the line between players (from any frame) that is MOST PARALLEL 
        to any table edge across all frames. Returns:
        - best_edge_idx: 0-3 (edge index in polygon)
        - best_frame: frame number where this occurred
        - min_angle_diff: smallest angular difference found
        """
        min_angle_diff = float('inf')
        self.best_edge_idx = -1
        self.best_frame = -1

        # Process all frames with both tracks and table data
        valid_frames = set(self.human_tracks.keys()).intersection(self.table_track.keys())
        for frame_idx in valid_frames:
            # Get player bounding boxes for this frame
            bbox0 = self.human_tracks[frame_idx][0]
            bbox1 = self.human_tracks[frame_idx][1]
            if not bbox0 or not bbox1:
                continue

            # Calculate centers of the two players
            cx0 = (bbox0[0] + bbox0[2]) / 2
            cy0 = (bbox0[1] + bbox0[3]) / 2
            cx1 = (bbox1[0] + bbox1[2]) / 2
            cy1 = (bbox1[1] + bbox1[3]) / 2

            # Calculate angle of the line between players
            dx = cx1 - cx0
            dy = cy1 - cy0
            player_angle = math.degrees(math.atan2(dy, dx))

            # Get table polygon points (handle both list and array formats)
            poly = self.table_track.get(frame_idx)
            if poly is None:
                continue

            # Check all 4 edges of the table polygon
            for edge_idx in range(4):
                pt1 = poly[edge_idx]
                pt2 = poly[(edge_idx + 1) % 4]

                # Calculate edge angle
                dx_edge = pt2[0] - pt1[0]
                dy_edge = pt2[1] - pt1[1]
                edge_angle = math.degrees(math.atan2(dy_edge, dx_edge))

                # Calculate angular difference (0°-90° range)
                diff = abs(player_angle - edge_angle) % 180
                angle_diff = min(diff, 180 - diff)

                # Update best match if this is the most parallel so far
                if angle_diff < min_angle_diff:
                    min_angle_diff = angle_diff
                    self.best_edge_idx = edge_idx
                    self.best_frame = frame_idx
        return {
            "edge_index": self.best_edge_idx,
            "frame": self.best_frame,
            "angle_diff": min_angle_diff,
            "points": (poly[self.best_edge_idx], poly[(self.best_edge_idx + 1) % 4]),
            "indices": (self.best_edge_idx, (self.best_edge_idx + 1) % 4),
            "tracked_ids": self.tracked_ids
        }
        
    def sort_points(self):
        """
        Re-orders the table_track polygons so that every frame's polygon is sorted 
        relative to the orientation defined by the best edge from best_frame.
        The desired order is: [top left, bottom left, top right, bottom right].

        This method rotates the best frame's polygon by the smallest angle that
        makes the best edge vertical (aligned with the y-axis) without flipping 
        the table horizontally. After rotation, the points are split into left and 
        right groups based on their x-values.
        """
        # Ensure find_most_parallel_line has been run.
        if self.best_edge_idx < 0 or self.best_frame < 0:
            print("Please run find_most_parallel_line first.")
            return

        # --- Helper function to rotate a point ---
        def rotate_point(pt, theta, origin=(0, 0)):
            ox, oy = origin
            px, py = pt
            qx = ox + math.cos(theta) * (px - ox) - math.sin(theta) * (py - oy)
            qy = oy + math.sin(theta) * (px - ox) + math.cos(theta) * (py - oy)
            return (qx, qy)

        # Get the best frame's polygon and validate its length.
        best_poly = self.table_track[self.best_frame]
        if len(best_poly) != 4:
            raise ValueError("Polygon in best_frame does not have 4 points.")

        # Get the endpoints of the best edge.
        p0 = best_poly[self.best_edge_idx]
        p1 = best_poly[(self.best_edge_idx + 1) % 4]

        # Compute the center of the polygon.
        center_x = sum(point[0] for point in best_poly) / 4
        center_y = sum(point[1] for point in best_poly) / 4
        center = (center_x, center_y)

        # Compute the angle of the best edge.
        edge_dx = p1[0] - p0[0]
        edge_dy = p1[1] - p0[1]
        edge_angle = math.atan2(edge_dy, edge_dx)

        # Determine the minimal rotation needed to make the edge vertical.
        # Vertical corresponds to angles of +pi/2 or -pi/2.
        candidate1 = math.pi/2 - edge_angle
        candidate2 = -math.pi/2 - edge_angle
        rot_angle = candidate1 if abs(candidate1) < abs(candidate2) else candidate2

        # Rotate all points in the best frame polygon.
        rotated_best = [rotate_point(pt, rot_angle, origin=center) for pt in best_poly]

        # After rotation, split the points into left and right halves using the x-value.
        indices = list(range(4))
        indices_sorted_by_x = sorted(indices, key=lambda i: rotated_best[i][0])
        left_indices = indices_sorted_by_x[:2]
        right_indices = indices_sorted_by_x[2:]

        # For each half, determine the top and bottom based on the y-value 
        # (smaller y means "higher" in image coordinates).
        top_left_idx = min(left_indices, key=lambda i: rotated_best[i][1])
        bottom_left_idx = max(left_indices, key=lambda i: rotated_best[i][1])
        top_right_idx = min(right_indices, key=lambda i: rotated_best[i][1])
        bottom_right_idx = max(right_indices, key=lambda i: rotated_best[i][1])

        # The desired ordering is: [top_left_idx, bottom_left_idx, bottom_right_idx, top_right_idx].
        permutation = [top_left_idx, bottom_left_idx, bottom_right_idx, top_right_idx]

        # Apply the same permutation to all frames in table_track.
        new_table_track = {}
        for frame_idx, poly in self.table_track.items():
            if len(poly) != 4:
                raise ValueError(f"Polygon in frame {frame_idx} does not have 4 points.")
            new_poly = [poly[i] for i in permutation]
            new_table_track[frame_idx] = new_poly

        return new_table_track