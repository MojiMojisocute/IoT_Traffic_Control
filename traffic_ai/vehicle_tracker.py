"""
🔍🔢 Vehicle Tracker + Counter for Smart Traffic System
ติดตามรถ + นับรถแต่ละทิศทาง (รวมกัน!)

Features:
- ✅ SORT tracking with Kalman Filter
- ✅ Track ID assignment
- ✅ Lane detection (เลนไหน/ทิศไหน)
- ✅ Counting with virtual lines
- ✅ Direction classification (North, South, East, West)
- ✅ Vehicle counting per direction
- ✅ Prevent double counting
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import logging
from collections import defaultdict, deque
from scipy.optimize import linear_sum_assignment
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KalmanBoxTracker:
    """
    🎯 Kalman Filter for tracking bounding boxes
    
    State: [x, y, s, r, vx, vy, vs, vr]
    - x, y: center position
    - s: scale (area)
    - r: aspect ratio
    - vx, vy, vs, vr: velocities
    """
    
    count = 0  # Track ID counter
    
    def __init__(self, bbox: List[int]):
        """
        Initialize Kalman tracker with bounding box
        
        Args:
            bbox: [x, y, w, h]
        """
        # ตั้งค่า Kalman Filter (constant velocity model)
        self.kf_x = np.zeros((8, 1), dtype=np.float32)
        self.kf_P = np.eye(8, dtype=np.float32) * 10
        self.kf_F = np.eye(8, dtype=np.float32)
        self.kf_H = np.zeros((4, 8), dtype=np.float32)
        self.kf_R = np.eye(4, dtype=np.float32) * 1
        self.kf_Q = np.eye(8, dtype=np.float32)
        
        # State transition
        for i in range(4):
            self.kf_F[i, i+4] = 1.0
        
        # Measurement matrix
        for i in range(4):
            self.kf_H[i, i] = 1.0
        
        # Process noise
        for i in range(4, 8):
            self.kf_Q[i, i] = 0.01
        
        # Initialize state
        x, y, w, h = bbox
        s = w * h
        r = w / float(h) if h != 0 else 1.0
        
        self.kf_x[0] = x + w/2
        self.kf_x[1] = y + h/2
        self.kf_x[2] = s
        self.kf_x[3] = r
        
        # Track info
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        
        # History
        self.history = deque(maxlen=30)
        self.history.append(bbox)
    
    def predict(self) -> np.ndarray:
        """Predict next state"""
        self.kf_x = self.kf_F @ self.kf_x
        self.kf_P = self.kf_F @ self.kf_P @ self.kf_F.T + self.kf_Q
        
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
        return self._state_to_bbox(self.kf_x)
    
    def update(self, bbox: List[int]):
        """Update with new measurement"""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        
        z = self._bbox_to_z(bbox)
        
        y = z - self.kf_H @ self.kf_x
        S = self.kf_H @ self.kf_P @ self.kf_H.T + self.kf_R
        K = self.kf_P @ self.kf_H.T @ np.linalg.inv(S)
        
        self.kf_x = self.kf_x + K @ y
        self.kf_P = (np.eye(8) - K @ self.kf_H) @ self.kf_P
        
        self.history.append(bbox)
    
    def get_state(self) -> List[int]:
        """Get current bbox"""
        return self._state_to_bbox(self.kf_x)
    
    def _bbox_to_z(self, bbox: List[int]) -> np.ndarray:
        """Convert bbox to measurement"""
        x, y, w, h = bbox
        s = w * h
        r = w / float(h) if h != 0 else 1.0
        return np.array([[x + w/2], [y + h/2], [s], [r]], dtype=np.float32)
    
    def _state_to_bbox(self, x: np.ndarray) -> List[int]:
        """Convert state to bbox"""
        cx, cy, s, r = x[0, 0], x[1, 0], x[2, 0], x[3, 0]
        w = np.sqrt(s * r)
        h = s / w if w != 0 else 1.0
        
        return [int(cx - w/2), int(cy - h/2), int(w), int(h)]


class VehicleTrackerCounter:
    """
    🚗🔢 SORT Tracker + Lane Counter รวมกัน!
    
    ทำหน้าที่:
    1. Track รถด้วย Kalman Filter + Hungarian
    2. แบ่ง lane (ทิศเหนือ/ใต้/ตะวันออก/ตะวันตก)
    3. นับรถที่ผ่านเส้น virtual line
    4. ป้องกันนับซ้ำ
    """
    
    def __init__(
        self,
        frame_width: int = 1280,
        frame_height: int = 720,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        counting_lines: Optional[Dict] = None
    ):
        """
        Initialize Tracker + Counter
        
        Args:
            frame_width: ความกว้างของเฟรม
            frame_height: ความสูงของเฟรม
            max_age: เก็บ track ที่หายไปกี่เฟรม
            min_hits: ต้องเจอกี่ครั้งถึงจะนับ
            iou_threshold: threshold สำหรับ matching
            counting_lines: กำหนดเส้นนับเอง (optional)
                {
                    'north': {'y': 200, 'direction': 'up'},
                    'south': {'y': 520, 'direction': 'down'},
                    'east': {'x': 960, 'direction': 'right'},
                    'west': {'x': 320, 'direction': 'left'}
                }
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        # Tracker
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0
        
        # Counter - เก็บว่า track ไหนนับไปแล้ว
        self.counted_tracks: Dict[str, Set[int]] = {
            'north': set(),
            'south': set(),
            'east': set(),
            'west': set()
        }
        
        # Vehicle counts per direction
        self.vehicle_counts = {
            'north': 0,
            'south': 0,
            'east': 0,
            'west': 0
        }
        
        # Counting lines (virtual lines)
        if counting_lines:
            self.counting_lines = counting_lines
        else:
            # Default: แบ่งเป็น 4 ทิศทาง
            self.counting_lines = {
                'north': {
                    'y': int(frame_height * 0.25),  # บน 1/4
                    'direction': 'up'  # รถวิ่งขึ้น
                },
                'south': {
                    'y': int(frame_height * 0.75),  # ล่าง 3/4
                    'direction': 'down'  # รถวิ่งลง
                },
                'east': {
                    'x': int(frame_width * 0.75),  # ขวา 3/4
                    'direction': 'right'  # รถวิ่งขวา
                },
                'west': {
                    'x': int(frame_width * 0.25),  # ซ้าย 1/4
                    'direction': 'left'  # รถวิ่งซ้าย
                }
            }
        
        # Statistics
        self.total_tracks = 0
        self.active_tracks = 0
        self.track_history = defaultdict(list)
    
    def update(self, detections: List[Dict]) -> Tuple[List[Dict], Dict[str, int]]:
        """
        Update tracks + นับรถ
        
        Args:
            detections: List of detections from YOLO
        
        Returns:
            (tracks, counts)
            - tracks: List of tracked vehicles with lane info
            - counts: Vehicle counts per direction
        """
        self.frame_count += 1
        
        # 1️⃣ Predict existing tracks
        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t] = pos
            
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        for t in reversed(to_del):
            self.trackers.pop(t)
        
        # 2️⃣ Match detections to tracks
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(
            detections, trks
        )
        
        # 3️⃣ Update matched tracks
        for m in matched:
            det_idx, trk_idx = m
            self.trackers[trk_idx].update(detections[det_idx]['bbox'])
        
        # 4️⃣ Create new tracks
        for i in unmatched_dets:
            trk = KalmanBoxTracker(detections[i]['bbox'])
            self.trackers.append(trk)
            self.total_tracks += 1
        
        # 5️⃣ Remove old tracks
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        
        # 6️⃣ สร้าง output + นับรถ
        tracks = []
        self.active_tracks = 0
        
        for trk in self.trackers:
            if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                bbox = trk.get_state()
                center = [bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2]
                
                # หา detection ที่ match
                det_info = {'class': 'car', 'confidence': 0.5}
                for m in matched:
                    if m[1] == self.trackers.index(trk):
                        det_info = {
                            'class': detections[m[0]]['class'],
                            'confidence': detections[m[0]]['confidence']
                        }
                        break
                
                # 🎯 หา lane/direction
                lane = self._get_lane(center, bbox)
                
                # 🔢 นับรถ (ถ้าผ่านเส้น)
                if lane:
                    self._count_vehicle(trk.id, center, lane, trk.history)
                
                tracks.append({
                    'track_id': trk.id,
                    'class': det_info['class'],
                    'confidence': det_info['confidence'],
                    'bbox': bbox,
                    'center': center,
                    'lane': lane,
                    'age': trk.age,
                    'hits': trk.hits
                })
                
                self.active_tracks += 1
                
                # บันทึก history
                self.track_history[trk.id].append({
                    'frame': self.frame_count,
                    'bbox': bbox,
                    'center': center,
                    'lane': lane
                })
        
        return tracks, self.vehicle_counts.copy()
    
    def _get_lane(self, center: List[int], bbox: List[int]) -> Optional[str]:
        """
        หาว่ารถอยู่เลนไหน (ทิศไหน)
        
        Args:
            center: [cx, cy]
            bbox: [x, y, w, h]
        
        Returns:
            'north', 'south', 'east', 'west' หรือ None
        """
        cx, cy = center
        
        # แบ่งตามตำแหน่งในเฟรม
        mid_x = self.frame_width // 2
        mid_y = self.frame_height // 2
        
        # แบ่งเป็น 4 quadrants
        if cy < mid_y:  # บนครึ่ง
            if cx < mid_x:
                return 'west'  # บนซ้าย → ทิศตะวันตก
            else:
                return 'north'  # บนขวา → ทิศเหนือ
        else:  # ล่างครึ่ง
            if cx < mid_x:
                return 'south'  # ล่างซ้าย → ทิศใต้
            else:
                return 'east'  # ล่างขวา → ทิศตะวันออก
    
    def _count_vehicle(
        self, 
        track_id: int, 
        center: List[int], 
        lane: str,
        history: deque
    ):
        """
        นับรถที่ผ่านเส้น virtual line
        
        Args:
            track_id: Track ID
            center: [cx, cy]
            lane: 'north', 'south', 'east', 'west'
            history: ประวัติตำแหน่ง
        """
        # ถ้านับไปแล้ว skip
        if track_id in self.counted_tracks[lane]:
            return
        
        # ต้องมี history อย่างน้อย 2 จุด
        if len(history) < 2:
            return
        
        cx, cy = center
        line = self.counting_lines.get(lane)
        
        if not line:
            return
        
        # ดึงตำแหน่งก่อนหน้า
        prev_bbox = history[-2]
        prev_cx = prev_bbox[0] + prev_bbox[2]//2
        prev_cy = prev_bbox[1] + prev_bbox[3]//2
        
        # ตรวจสอบว่าข้ามเส้นหรือไม่
        crossed = False
        
        if 'y' in line:  # เส้นนอน (north/south)
            line_y = line['y']
            
            if line['direction'] == 'up':
                # รถวิ่งขึ้น (ทิศเหนือ)
                if prev_cy > line_y and cy <= line_y:
                    crossed = True
            else:  # down
                # รถวิ่งลง (ทิศใต้)
                if prev_cy < line_y and cy >= line_y:
                    crossed = True
        
        elif 'x' in line:  # เส้นตั้ง (east/west)
            line_x = line['x']
            
            if line['direction'] == 'right':
                # รถวิ่งขวา (ทิศตะวันออก)
                if prev_cx < line_x and cx >= line_x:
                    crossed = True
            else:  # left
                # รถวิ่งซ้าย (ทิศตะวันตก)
                if prev_cx > line_x and cx <= line_x:
                    crossed = True
        
        # ถ้าข้ามเส้น → นับ!
        if crossed:
            self.vehicle_counts[lane] += 1
            self.counted_tracks[lane].add(track_id)
            logger.info(f"🚗 Counted: Track #{track_id} in lane '{lane}' → Total: {self.vehicle_counts[lane]}")
    
    def _associate_detections_to_trackers(
        self, 
        detections: List[Dict], 
        trackers: np.ndarray
    ) -> Tuple[List, List[int], List[int]]:
        """จับคู่ detections กับ trackers"""
        if len(trackers) == 0:
            return [], list(range(len(detections))), []
        
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = self._iou(det['bbox'], trk)
        
        if min(iou_matrix.shape) > 0:
            cost_matrix = 1 - iou_matrix
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matched_indices = np.array(list(zip(row_ind, col_ind)))
        else:
            matched_indices = np.empty(shape=(0, 2))
        
        unmatched_detections = []
        for d in range(len(detections)):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        
        unmatched_trackers = []
        for t in range(len(trackers)):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)
        
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        
        return matches, unmatched_detections, unmatched_trackers
    
    def _iou(self, bbox1: List[int], bbox2: np.ndarray) -> float:
        """คำนวณ IoU"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def get_stats(self) -> Dict:
        """ดึงสถิติ"""
        return {
            'frame_count': self.frame_count,
            'total_tracks': self.total_tracks,
            'active_tracks': self.active_tracks,
            'vehicle_counts': self.vehicle_counts.copy(),
            'total_counted': sum(self.vehicle_counts.values())
        }
    
    def reset_counts(self):
        """รีเซ็ตตัวนับ (ไม่รีเซ็ต tracker)"""
        self.vehicle_counts = {
            'north': 0,
            'south': 0,
            'east': 0,
            'west': 0
        }
        self.counted_tracks = {
            'north': set(),
            'south': set(),
            'east': set(),
            'west': set()
        }
        logger.info("🔄 Vehicle counts reset")
    
    def reset(self):
        """รีเซ็ตทั้งหมด"""
        self.trackers = []
        self.frame_count = 0
        self.total_tracks = 0
        self.active_tracks = 0
        self.track_history.clear()
        self.reset_counts()
        KalmanBoxTracker.count = 0
        logger.info("🔄 Full reset complete")


# ==========================================
# 🧪 ตัวอย่างการใช้งาน
# ==========================================

if __name__ == "__main__":
    print("🔍🔢 Vehicle Tracker + Counter Test")
    print("=" * 60)
    
    # สร้าง tracker+counter
    tracker = VehicleTrackerCounter(
        frame_width=1280,
        frame_height=720,
        max_age=30,
        min_hits=3
    )
    
    print(f"✅ Tracker+Counter initialized")
    print(f"\n📏 Counting Lines:")
    for lane, line in tracker.counting_lines.items():
        print(f"   {lane}: {line}")
    
    # จำลองรถวิ่งผ่าน
    print("\n🚗 Simulating vehicles...")
    
    # รถทิศเหนือ (วิ่งขึ้น)
    for i in range(5):
        y = 300 - (i * 20)  # ค่อยๆ ขยับขึ้น
        detections = [{
            'class': 'car',
            'confidence': 0.9,
            'bbox': [700, y, 50, 80],
            'center': [725, y+40]
        }]
        tracks, counts = tracker.update(detections)
        
    print(f"\n📊 North count: {counts['north']}")
    
    # รถทิศใต้ (วิ่งลง)
    for i in range(3):
        y = 400 + (i * 20)
        detections = [{
            'class': 'bus',
            'confidence': 0.88,
            'bbox': [300, y, 60, 90],
            'center': [330, y+45]
        }]
        tracks, counts = tracker.update(detections)
    
    print(f"📊 South count: {counts['south']}")
    
    # แสดงสถิติรวม
    print(f"\n📊 Final Stats:")
    stats = tracker.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n✅ Test Complete!")
    print("\n💡 ใช้ร่วมกับ Camera + YOLO:")
    print("   from camera import Camera")
    print("   from yolo_detector import YOLODetector")
    print("   from vehicle_tracker import VehicleTrackerCounter")
    print("   ")
    print("   camera = Camera(source=0)")
    print("   detector = YOLODetector()")
    print("   tracker = VehicleTrackerCounter()")
    print("   ")
    print("   while True:")
    print("       frame = camera.get_frame()")
    print("       detections = detector.detect(frame)")
    print("       tracks, counts = tracker.update(detections)")
    print("       print(f'Counts: {counts}')")