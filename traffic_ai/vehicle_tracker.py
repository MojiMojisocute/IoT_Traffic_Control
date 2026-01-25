import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import logging
from collections import defaultdict, deque
from scipy.optimize import linear_sum_assignment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KalmanBoxTracker:
    count = 0
    
    def __init__(self, bbox: List[int]):
        self.kf_x = np.zeros((8, 1), dtype=np.float32)
        self.kf_P = np.eye(8, dtype=np.float32) * 10
        self.kf_F = np.eye(8, dtype=np.float32)
        self.kf_H = np.zeros((4, 8), dtype=np.float32)
        self.kf_R = np.eye(4, dtype=np.float32)
        self.kf_Q = np.eye(8, dtype=np.float32)
        
        for i in range(4):
            self.kf_F[i, i+4] = 1.0
            self.kf_H[i, i] = 1.0
        
        for i in range(4, 8):
            self.kf_Q[i, i] = 0.01
        
        x, y, w, h = bbox
        s = w * h
        r = w / float(h) if h != 0 else 1.0
        
        self.kf_x[0] = x + w/2
        self.kf_x[1] = y + h/2
        self.kf_x[2] = s
        self.kf_x[3] = r
        
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        
        self.history = deque(maxlen=30)
        self.history.append(bbox)
    
    def predict(self) -> np.ndarray:
        self.kf_x = self.kf_F @ self.kf_x
        self.kf_P = self.kf_F @ self.kf_P @ self.kf_F.T + self.kf_Q
        
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
        return self._state_to_bbox(self.kf_x)
    
    def update(self, bbox: List[int]):
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
        return self._state_to_bbox(self.kf_x)
    
    def _bbox_to_z(self, bbox: List[int]) -> np.ndarray:
        x, y, w, h = bbox
        s = w * h
        r = w / float(h) if h != 0 else 1.0
        return np.array([[x + w/2], [y + h/2], [s], [r]], dtype=np.float32)
    
    def _state_to_bbox(self, x: np.ndarray) -> List[int]:
        cx, cy, s, r = x[0, 0], x[1, 0], x[2, 0], x[3, 0]
        
        if np.isnan(cx) or np.isnan(cy) or np.isnan(s) or np.isnan(r):
            return [0, 0, 1, 1]
        
        if s <= 0 or r <= 0:
            return [0, 0, 1, 1]
        
        w = np.sqrt(s * r)
        h = s / w if w != 0 else 1.0
        
        x_pos = max(0, int(cx - w/2))
        y_pos = max(0, int(cy - h/2))
        width = max(1, int(w))
        height = max(1, int(h))
        
        return [x_pos, y_pos, width, height]


class VehicleTrackerCounter:
    def __init__(
        self,
        frame_width: int = 1280,
        frame_height: int = 720,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        counting_lines: Optional[Dict] = None
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0
        
        self.counted_tracks: Dict[str, Set[int]] = {
            'north': set(), 'south': set(), 'east': set(), 'west': set()
        }
        
        self.vehicle_counts = {
            'north': 0, 'south': 0, 'east': 0, 'west': 0
        }
        
        if counting_lines:
            self.counting_lines = counting_lines
        else:
            self.counting_lines = {
                'north': {'y': int(frame_height * 0.25), 'direction': 'up'},
                'south': {'y': int(frame_height * 0.75), 'direction': 'down'},
                'east': {'x': int(frame_width * 0.75), 'direction': 'right'},
                'west': {'x': int(frame_width * 0.25), 'direction': 'left'}
            }
        
        self.total_tracks = 0
        self.active_tracks = 0
        self.track_history = defaultdict(list)
    
    def update(self, detections: List[Dict]) -> Tuple[List[Dict], Dict[str, int]]:
        self.frame_count += 1
        
        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trk[:] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        
        for t in reversed(to_del):
            self.trackers.pop(t)
        
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(
            detections, trks
        )
        
        for m in matched:
            self.trackers[m[1]].update(detections[m[0]]['bbox'])
        
        for i in unmatched_dets:
            trk = KalmanBoxTracker(detections[i]['bbox'])
            self.trackers.append(trk)
            self.total_tracks += 1
        
        tracks = []
        self.active_tracks = 0
        
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            
            if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                bbox = d
                center = [bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2]
                
                lane = self._get_lane(center, bbox)
                
                if lane:
                    self._count_vehicle(trk.id, center, lane, trk.history)
                
                self.active_tracks += 1
                
                tracks.append({
                    'track_id': trk.id,
                    'class': 'vehicle',
                    'confidence': 0.9,
                    'frame': self.frame_count,
                    'bbox': bbox,
                    'center': center,
                    'lane': lane
                })
            
            i -= 1
            
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        
        if len(tracks) > 0:
            for track in tracks:
                self.track_history[track['track_id']].append({
                    'frame': self.frame_count,
                    'bbox': track['bbox'],
                    'center': track['center'],
                    'lane': track['lane']
                })
        
        return tracks, self.vehicle_counts.copy()
    
    def _get_lane(self, center: List[int], bbox: List[int]) -> Optional[str]:
        cx, cy = center
        
        mid_x = self.frame_width // 2
        mid_y = self.frame_height // 2
        
        if cy < mid_y:
            if cx < mid_x:
                return 'west'
            else:
                return 'north'
        else:
            if cx < mid_x:
                return 'south'
            else:
                return 'east'
    
    def _count_vehicle(self, track_id: int, center: List[int], lane: str, history: deque):
        if track_id in self.counted_tracks[lane]:
            return
        
        if len(history) < 2:
            return
        
        cx, cy = center
        line = self.counting_lines.get(lane)
        
        if not line:
            return
        
        prev_bbox = history[-2]
        prev_cx = prev_bbox[0] + prev_bbox[2]//2
        prev_cy = prev_bbox[1] + prev_bbox[3]//2
        
        crossed = False
        
        if 'y' in line:
            line_y = line['y']
            
            if line['direction'] == 'up':
                if prev_cy > line_y and cy <= line_y:
                    crossed = True
            else:
                if prev_cy < line_y and cy >= line_y:
                    crossed = True
        
        elif 'x' in line:
            line_x = line['x']
            
            if line['direction'] == 'right':
                if prev_cx < line_x and cx >= line_x:
                    crossed = True
            else:
                if prev_cx > line_x and cx <= line_x:
                    crossed = True
        
        if crossed:
            self.vehicle_counts[lane] += 1
            self.counted_tracks[lane].add(track_id)
    
    def _associate_detections_to_trackers(
        self, 
        detections: List[Dict], 
        trackers: np.ndarray
    ) -> Tuple[List, List[int], List[int]]:
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
        return {
            'frame_count': self.frame_count,
            'total_tracks': self.total_tracks,
            'active_tracks': self.active_tracks,
            'vehicle_counts': self.vehicle_counts.copy(),
            'total_counted': sum(self.vehicle_counts.values())
        }
    
    def reset_counts(self):
        self.vehicle_counts = {'north': 0, 'south': 0, 'east': 0, 'west': 0}
        self.counted_tracks = {'north': set(), 'south': set(), 'east': set(), 'west': set()}
    
    def reset(self):
        self.trackers = []
        self.frame_count = 0
        self.total_tracks = 0
        self.active_tracks = 0
        self.track_history.clear()
        self.reset_counts()
        KalmanBoxTracker.count = 0