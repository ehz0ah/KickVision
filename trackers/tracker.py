from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import cv2
import sys
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    # Detect objects in the frames
    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        # Split the frames into batches
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            batch_detections = self.model.predict(batch_frames, conf = 0.1)
            detections += batch_detections
        return detections

    def get_object_track(self, frames, read_from_stubs=False, stub_path=None):
        # If read_from_stubs is True, read the tracks from the stub file
        if read_from_stubs and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks
        # Detect objects in the frames
        detections = self.detect_frames(frames)

        tracks = {
            "players" : [], 
            "referees" : [],
            "ball" : []
        }

        # Convert the detections to supervision format
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names   # Original {0:Person, 1:Car, 2:Motorcycle}
            cls_names_inv = {value:key for key, value in cls_names.items()}  # Inverted {'Person':0, 'Car':1, 'Motorcycle':2}

            # Convert to supervision format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert Goalkeeper to Player (For simplicity)
            for object_index, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[object_index] = cls_names_inv['player']

            # Track the objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            
            # Append a dictionary where key is the track_id and value is the bounding box
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

                # No need to do for ball since there is only one ball in each frame

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}  # Only one ball

        # Save the tracks to a file
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])

        x_center, y2_center = get_center_of_bbox(bbox)  # Use y2 instead of y2_center to draw the ellipse at the bottom of the bbox
        width = get_bbox_width(bbox) # One of the 2 radius for the ellipse (There is major and minor axis for an ellipse)
        
        cv2.ellipse(
            frame,
            center = (x_center, y2),
            axes = (int(width), int(0.35*width)),  # Major and minor radius
            angle = 0.0,
            startAngle = -45,
            endAngle = 235,
            color = color,
            thickness = 2,
            lineType = cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - (rectangle_height // 2)) + 15
        y2_rect = (y2 + (rectangle_height // 2)) + 15

        if track_id is not None:
            cv2.rectangle(frame, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), color, cv2.FILLED)
            
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(frame, str(track_id), (int(x1_text), int(y2_rect - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return frame


    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, y_center = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])

        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2)

        return frame



    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            # print(f"Processing frame {frame_num}...")
            # Make a copy of the frame so we don't modify the original frame
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw the players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

            # Draw the referees
            for track_id, referee in referee_dict.items():
                # print(f"Drawing ellipse for player {track_id} on frame {frame_num}.")
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255), track_id)

            # Draw the ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            output_video_frames.append(frame)
        # print("Finished drawing all frames.")
        return output_video_frames
