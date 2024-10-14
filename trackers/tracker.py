from ultralytics import YOLO
import supervision as sv
import pickle
import os

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
