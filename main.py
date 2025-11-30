import cv2  
import torch
import numpy as np
from ultralytics import YOLO
from boxmot.tracker_zoo import create_tracker
from pathlib import Path
import time
import torch.nn.functional as F
import torchreid
from torchvision import transforms as T
from collections import deque

class AirportSurveillanceSystem:
    def __init__(self):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        self.UNATTENDED_TIME_THRESHOLD = 1.0
        self.IOU_THRESHOLD = 0.01 # Reverted to 0.3 for better bag-person association
        self.REID_SIMILARITY_THRESHOLD = 0.7

        self.PERSON_CLASS_ID = 0
        self.LUGGAGE_CLASS_IDS = [24, 26, 28]
        self.REID_INPUT_SIZE = (128, 256)

        self.VIDEO_PATH = "C:/Users/Lenovo/Downloads/My Movie2.mp4/My Movie2.mp4"
        
        # --- CORRECTED PATH TO YOUR MANUALLY DOWNLOADED RE-ID MODEL (.pth source) ---
        # Fixed the 'lr0.0015' part
        self.REID_SOURCE_PATH_FOR_TORCHREID = Path("C:/Users/Lenovo/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth")
        
        # This is the path where the BoxMOT-compatible .pt file will be saved
        # This path remains in the cache as previously intended.
        self.REID_MODEL_FOR_BOXMOT_PATH = Path("C:/Users/Lenovo/.cache/torch/checkpoints/osnet_x1_0_market1501_for_boxmot.pt")
        
        # --- PATH TO BOTSORT.YAML ---
        self.TRACKING_CONFIG_PATH = Path("C:/Users/Lenovo/boxmot/boxmot/configs/trackers/botsort.yaml") 

        # Load and prepare the Re-ID model for BoxMOT compatibility
        self.reid_model_for_feature_extraction = self._load_reid_model() 

        self.yolo_model = YOLO('yolov8l.pt').to(self.device)
        self.yolo_model.fuse()

        # --- Using 'botsort' ---
        self.tracker = create_tracker('botsort', self.TRACKING_CONFIG_PATH, reid_weights=self.REID_MODEL_FOR_BOXMOT_PATH)

        self.reid_transform = T.Compose([
            T.ToPILImage(),
            T.Resize(self.REID_INPUT_SIZE),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.person_reid_features = {}
        self.bag_state = {}

        self.TRACKLET_LENGTH = 10  # Store last 10 features for stable ID
        # Persistent gallery: {global_id: deque of last N features}
        self.person_gallery = {}
        # Mapping from transient BoxMOT IDs to long-term global IDs
        self.boxmot_to_global_id = {}
        self.next_global_id = 1
        self.current_frame_person_features = {}
        self.theft_incidents = []


    def extract_reid_feature(self, frame, box):
        x1, y1, x2, y2 = box
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or x2 - x1 < 1 or y2 - y1 < 1:
            return None

        try:
            feat = self.reid_model_for_feature_extraction(self.reid_transform(crop).unsqueeze(0).to(self.device))
            feat = F.normalize(feat, p=2, dim=1)
            return feat.squeeze(0).cpu()
        except Exception as e:
            return None
        
    def match_reid_feature_to_gallery(self, feat):
        best_id = None
        best_sim = 0
        for gid, feats in self.person_gallery.items():
            if not feats: continue
            sims = [F.cosine_similarity(feat.unsqueeze(0), f.unsqueeze(0)).item() for f in feats]
            max_sim = max(sims)

            if max_sim > best_sim:
                best_sim = max_sim
                best_id = gid

        if best_sim >= self.REID_SIMILARITY_THRESHOLD:
            self.person_gallery[best_id].append(feat)
            return best_id
        else:
            new_id = self.next_global_id
            self.person_gallery[new_id] = deque([feat], maxlen=self.TRACKLET_LENGTH)
            self.next_global_id += 1
            return new_id

    def _load_reid_model(self):
        """
        Loads the Re-ID model from the specified .pth file using torchreid,
        and then saves its state_dict as a .pt file for BoxMOT compatibility.
        """
        print("Attempting to load/process Re-ID model for feature extraction and BoxMOT compatibility...")

        # Build the OSNet model structure
        model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=751, # Market1501 has 751 identities
            loss='softmax',
            pretrained=False # Set to False as we are loading from a local file, not auto-downloading
        ).to(self.device)

        # Check if the source .pth file exists
        if not self.REID_SOURCE_PATH_FOR_TORCHREID.is_file():
            print(f"ERROR: Re-ID source file not found at {self.REID_SOURCE_PATH_FOR_TORCHREID}.")
            print("Please ensure you have manually downloaded 'osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth' and placed it there.")
            exit()
        
        print(f"Loading Re-ID weights from: {self.REID_SOURCE_PATH_FOR_TORCHREID}")
        try:
            # Load the state_dict from the .pth file
            state_dict = torch.load(self.REID_SOURCE_PATH_FOR_TORCHREID, map_location=self.device)
            
            # Extract 'state_dict' if the checkpoint wraps it (common in torchreid)
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            # Remove 'module.' prefix if the model was saved with DataParallel
            new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
            
            # Load the cleaned state_dict into the model
            model.load_state_dict(new_state_dict, strict=False)
            print("Successfully loaded Re-ID weights into torchreid model.")
        except Exception as e:
            print(f"Error loading Re-ID weights from {self.REID_SOURCE_PATH_FOR_TORCHREID}: {e}")
            exit()

        # Save the loaded model's state_dict to a .pt format for BoxMOT compatibility
        print(f"Saving Re-ID model to {self.REID_MODEL_FOR_BOXMOT_PATH} for BoxMOT compatibility...")
        try:
            torch.save(model.state_dict(), self.REID_MODEL_FOR_BOXMOT_PATH)
            print("Model successfully saved in .pt format for BoxMOT.")
        except Exception as e:
            print(f"Error saving model for BoxMOT: {e}")
            print("Please ensure you have write permissions to the cache directory.")
            exit() 

        # Final check if the BoxMOT compatible file exists
        if not self.REID_MODEL_FOR_BOXMOT_PATH.is_file():
            print(f"ERROR: BoxMOT compatible Re-ID weights file not found at {self.REID_MODEL_FOR_BOXMOT_PATH} AFTER saving attempt.")
            exit()
        else:
            print(f"BoxMOT Re-ID weights confirmed at: {self.REID_MODEL_FOR_BOXMOT_PATH}")

        model.eval() # Set the model to evaluation mode
        return model

    def compute_iou(self, boxA, boxB):
        """Calculates the Intersection Over Union (IOU) of two bounding boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    def process_video(self):
        """Processes the video feed for person and baggage tracking."""
        cap = cv2.VideoCapture(self.VIDEO_PATH)

        # Get video info
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
        fps = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Output file to save processed video
        out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

        if not cap.isOpened():
            print("Error: Could not open video source.")
            return

        while cap.isOpened():
            fps_start = time.time()  # Start timer for FPS calculation
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()
            persons, bags = self.detect_and_track(frame)
            self.update_states(persons, bags, current_time)
            self.draw(frame, persons, bags, current_time)

            fps_end = time.time()  # End timer
            fps = 1 / (fps_end - fps_start + 1e-6)  # Calculate FPS

            # Display FPS on the frame
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            # Save the processed frame to output video
            out.write(frame)

            # Show the frame live
            cv2.imshow("Airport Surveillance", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()   # release the video writer
        cv2.destroyAllWindows()


    def detect_and_track(self, frame):
        results = self.yolo_model(frame, verbose=False)[0]

        detections = []
        for box_data in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = box_data
            cls_int = int(cls)
            if cls_int == self.PERSON_CLASS_ID or cls_int in self.LUGGAGE_CLASS_IDS:
                detections.append([x1, y1, x2, y2, conf, cls_int])

        detections_np = np.array(detections) if detections else np.empty((0, 6))
        tracks = self.tracker.update(detections_np, frame)

        persons, bags = [], []
        current_frame_boxmot_ids = set()

        for t in tracks:
            x1, y1, x2, y2 = map(int, t[:4])
            boxmot_id = int(t[4])
            cls = int(t[6])
            bbox = (x1, y1, x2, y2)

            if cls == self.PERSON_CLASS_ID:
                current_frame_boxmot_ids.add(boxmot_id)

                # --- Extract feature properly ---
                feat = self.extract_reid_feature(frame, bbox)   # <- instead of using t[7:]
                if feat is None:
                    continue

                global_id = self.boxmot_to_global_id.get(boxmot_id)
                if global_id is None:
                    global_id = self.match_reid_feature_to_gallery(feat)
                    self.boxmot_to_global_id[boxmot_id] = global_id
                else:
                    self.person_gallery[global_id].append(feat)

                # 🚨 Store latest feature for theft check
                self.person_reid_features[global_id] = feat

                persons.append((global_id, bbox, feat))

            elif cls in self.LUGGAGE_CLASS_IDS:
                bags.append((boxmot_id, bbox))

        # Clean stale IDs
        stale_ids = list(self.boxmot_to_global_id.keys() - current_frame_boxmot_ids)
        for old_id in stale_ids:
            del self.boxmot_to_global_id[old_id]

        return persons, bags

    
    def compute_center(self, box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def euclidean_distance(self, p1, p2):
        return ((p1[0] - p2[0]) * 2 + (p1[1] - p2[1]) * 2) ** 0.5
    
    def get_bbox_for_id(self, person_id, persons):
        for pid, bbox, _ in persons:
            if pid == person_id:
                return bbox
        return None

    def update_states(self, persons, bags, current_time):
        for bag_tid, bag_bbox in bags:
            bag_info = self.bag_state.get(bag_tid)
            historical_owner = bag_info.get('owner_id') if bag_info else None
            historical_feat = bag_info.get('owner_reid_feat') if bag_info else None
            if bag_info:
                # Reset status each frame before re-checking
                bag_info['theft_flag'] = False
                bag_info['theft_by'] = None
                bag_info['unattended_flag'] = True  # default back to unattended unless proven otherwise


            # --- Step 1: Is historical owner still near? ---
            owner_is_near = False
            if historical_owner is not None:
                for pid, pbbox, _ in persons:
                    if pid == historical_owner:
                        iou = self.compute_iou(bag_bbox, pbbox)
                        if iou > self.IOU_THRESHOLD:
                            owner_is_near = True
                            bag_info['last_attended_time'] = current_time
                            bag_info['unattended_flag'] = False
                            bag_info['theft_flag'] = False
                            bag_info['theft_by'] = None
                            break

            if owner_is_near:
                continue  # Owner is close — bag is attended

            # --- Step 2: Is a new person close to the bag? ---
            best_iou = 0
            new_person_id = None
            new_person_feat = None
            for pid, pbbox, _ in persons:
                iou = self.compute_iou(bag_bbox, pbbox)
                if iou > self.IOU_THRESHOLD and iou > best_iou:
                    best_iou = iou
                    new_person_id = pid
                    new_person_feat = self.person_reid_features.get(pid)

            if new_person_id is not None:
                # 🚨 Ensure bag_info is initialized
                if bag_info is None:
                    self.bag_state[bag_tid] = {
                        'owner_id': None,
                        'owner_reid_feat': None,
                        'last_attended_time': current_time,
                        'unattended_flag': False,
                        'theft_flag': False,
                        'theft_by': None
                    }
                    bag_info = self.bag_state[bag_tid]

                # 🚨 Step 3: If bag had a historical owner, compare Re-ID
                if historical_feat is not None and new_person_feat is not None:
                    sim = F.cosine_similarity(new_person_feat.unsqueeze(0), historical_feat.unsqueeze(0)).item()

                    if sim < self.REID_SIMILARITY_THRESHOLD:
                        # 🧠 Is historical owner still close to the new person?
                        for pid, pbbox, _ in persons:
                            if pid == historical_owner:
                                overlap = self.compute_iou(pbbox, self.get_bbox_for_id(new_person_id, persons))
                                if overlap > self.IOU_THRESHOLD:
                                    # Still under owner's watch → do nothing
                                    bag_info['last_attended_time'] = current_time
                                    continue

                        # 🚨 Theft: Someone else picked it up and owner isn't near
                        print(f"⚠️ Potential Theft: Bag {bag_tid} picked up by P:{new_person_id} (Owner: P:{historical_owner})")
                        bag_info['unattended_flag'] = False   # not unattended, someone is holding it
                        bag_info['theft_flag'] = True
                        bag_info['theft_by'] = new_person_id
                        bag_info['last_attended_time'] = current_time

                else:
                    # Initial assignment — assign owner and reid
                    bag_info['owner_id'] = new_person_id
                    bag_info['owner_reid_feat'] = new_person_feat
                    bag_info['last_attended_time'] = current_time

            # --- Step 4: No one near bag and owner is gone for too long ---
            if bag_info:
                if current_time - bag_info['last_attended_time'] > self.UNATTENDED_TIME_THRESHOLD:
                    bag_info['unattended_flag'] = True
                    bag_info['theft_flag'] = False      # clear theft if no one is near
                    bag_info['theft_by'] = None


    def draw(self, frame, persons, bags, current_time):
        """Draws bounding boxes and labels on the frame."""
        # Draw persons
        for pid, bbox, _ in persons:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"P:{pid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Draw bags and unattended/theft alerts
        for bid, bbox in bags:
            x1, y1, x2, y2 = bbox
            bag_info = self.bag_state.get(bid)
            color = (0, 255, 0)  # Default: green for attended/normal
            text = f"Bag:{bid}"

            if bag_info and bag_info['theft_flag']:  # Check for theft first
                color = (0, 0, 255)  # Red for theft
                owner = bag_info['owner_id'] or "Unknown"
                unattended_time = current_time - bag_info['last_attended_time']
                text = f"THEFT:{bid} by P:{bag_info['theft_by']} (Owner: {owner})"
                # Display prominent theft warning

            elif bag_info and bag_info['unattended_flag']:  # Then check for general unattended state
                color = (0, 165, 255)  # Orange for unattended
                owner = bag_info['owner_id'] or "Unknown"
                unattended_time = current_time - bag_info['last_attended_time']
                text = f"UNATTENDED:{bid} ({owner}) {unattended_time:.1f}s"
                # Display prominent unattended luggage warning
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, text, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


if __name__ == "__main__":
    system = AirportSurveillanceSystem()
    system.process_video()
