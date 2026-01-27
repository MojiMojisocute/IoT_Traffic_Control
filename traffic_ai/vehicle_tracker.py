import numpy as np
import cv2
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
        
        self.counting_lines = self._normalize_counting_lines(counting_lines)
        
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0
        
        self.counted_tracks = {}
        self.vehicle_counts = {}
        
        if self.counting_lines:
            for lane_name in self.counting_lines.keys():
                self.counted_tracks[lane_name] = set()
                self.vehicle_counts[lane_name] = 0
        
        self.total_tracks = 0
        self.active_tracks = 0
        self.track_history = defaultdict(lambda: deque(maxlen=30))
    
    def _normalize_counting_lines(self, counting_lines):
        if not counting_lines or not isinstance(counting_lines, dict):
            return {}
        
        normalized = {}
        for name, data in counting_lines.items():
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], (list, tuple)) and len(data[0]) == 2:
                    normalized[name] = data
                elif isinstance(data, dict) and 'x1' in data:
                    normalized[name] = [(data['x1'], data['y1']), (data['x2'], data['y2'])]
            elif isinstance(data, dict):
                if 'x1' in data and 'y1' in data and 'x2' in data and 'y2' in data:
                    normalized[name] = [(data['x1'], data['y1']), (data['x2'], data['y2'])]
        
        return normalized
    
    def update(self, detections: List[Dict]) -> Tuple[List[Dict], Dict]:
        self.frame_count += 1
        
        if self.counting_lines:
            detections = [det for det in detections if self._is_in_any_polygon(det['center'])]
        
        if not detections:
            for trk in self.trackers:
                trk.predict()
            
            self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
            self.active_tracks = len([t for t in self.trackers if t.time_since_update < 1])
            return [], self.vehicle_counts.copy()
        
        predictions = []
        valid_trackers = []
        
        for trk in self.trackers:
            pos = trk.predict()
            if not np.any(np.isnan(pos)):
                predictions.append(pos)
                valid_trackers.append(trk)
        
        self.trackers = valid_trackers
        trks = np.array(predictions) if predictions else np.empty((0, 4))
        
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(
            detections, trks
        )
        
        for det_idx, trk_idx in matched:
            self.trackers[trk_idx].update(detections[det_idx]['bbox'])
        
        for i in unmatched_dets:
            trk = KalmanBoxTracker(detections[i]['bbox'])
            self.trackers.append(trk)
            self.total_tracks += 1
        
        tracks = []
        self.active_tracks = 0
        
        for trk in self.trackers[:]:
            if trk.time_since_update > self.max_age:
                continue
                
            if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                bbox = trk.get_state()
                center = [bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2]
                
                if self.counting_lines:
                    self._check_polygon_crossing(trk.id, center, trk.history)
                
                self.active_tracks += 1
                
                track_data = {
                    'track_id': trk.id,
                    'class': 'vehicle',
                    'confidence': 0.9,
                    'frame': self.frame_count,
                    'bbox': bbox,
                    'center': center
                }
                tracks.append(track_data)
                self.track_history[trk.id].append({
                    'frame': self.frame_count,
                    'bbox': bbox,
                    'center': center
                })
        
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
        
        return tracks, self.vehicle_counts.copy()
    
    def _is_in_any_polygon(self, center: List[int]) -> bool:
        if not self.counting_lines:
            return True
        
        x, y = center
        for polygon_points in self.counting_lines.values():
            if self._point_in_polygon(x, y, polygon_points):
                return True
        return False
    
    def _check_polygon_crossing(self, track_id: int, center: List[int], history: deque):
        if len(history) < 2:
            return
        
        prev_bbox = history[-2]
        prev_cx = prev_bbox[0] + prev_bbox[2] // 2
        prev_cy = prev_bbox[1] + prev_bbox[3] // 2
        
        curr_cx, curr_cy = center
        
        for lane_name, polygon_points in self.counting_lines.items():
            if track_id in self.counted_tracks[lane_name]:
                continue
            
            if not polygon_points or len(polygon_points) < 3:
                continue
            
            prev_inside = self._point_in_polygon(prev_cx, prev_cy, polygon_points)
            curr_inside = self._point_in_polygon(curr_cx, curr_cy, polygon_points)
            
            if prev_inside != curr_inside:
                self.vehicle_counts[lane_name] += 1
                self.counted_tracks[lane_name].add(track_id)
    
    def _point_in_polygon(self, x: float, y: float, polygon: List) -> bool:
        try:
            pts = np.array(polygon, np.int32)
            result = cv2.pointPolygonTest(pts, (int(x), int(y)), False)
            return result >= 0
        except:
            return False
    
    def _associate_detections_to_trackers(
        self, 
        detections: List[Dict], 
        trackers: np.ndarray
    ) -> Tuple[np.ndarray, List[int], List[int]]:
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), list(range(len(detections))), []
        
        if len(detections) == 0:
            return np.empty((0, 2), dtype=int), [], list(range(len(trackers)))
        
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        
        det_boxes = np.array([det['bbox'] for det in detections])
        
        for t in range(len(trackers)):
            iou_matrix[:, t] = self._batch_iou(det_boxes, trackers[t])
        
        cost_matrix = 1 - iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched_mask = iou_matrix[row_ind, col_ind] >= self.iou_threshold
        matched_indices = np.column_stack([row_ind[matched_mask], col_ind[matched_mask]])
        
        unmatched_detections = [d for d in range(len(detections)) if d not in matched_indices[:, 0]]
        unmatched_trackers = [t for t in range(len(trackers)) if t not in matched_indices[:, 1]]
        
        return matched_indices, unmatched_detections, unmatched_trackers
    
    def _batch_iou(self, bboxes1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
        x1 = bboxes1[:, 0].astype(np.float32)
        y1 = bboxes1[:, 1].astype(np.float32)
        w1 = bboxes1[:, 2].astype(np.float32)
        h1 = bboxes1[:, 3].astype(np.float32)
        
        x2, y2, w2, h2 = bbox2.astype(np.float32)
        
        x_left = np.maximum(x1, x2)
        y_top = np.maximum(y1, y2)
        x_right = np.minimum(x1 + w1, x2 + w2)
        y_bottom = np.minimum(y1 + h1, y2 + h2)
        
        intersection = np.maximum(0, x_right - x_left) * np.maximum(0, y_bottom - y_top)
        
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        iou = np.zeros_like(intersection, dtype=np.float32)
        mask = union > 0
        iou[mask] = intersection[mask] / union[mask]
        
        return iou
    
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
        for lane_name in self.counted_tracks.keys():
            self.vehicle_counts[lane_name] = 0
            self.counted_tracks[lane_name] = set()
    
    def reset(self):
        self.trackers = []
        self.frame_count = 0
        self.total_tracks = 0
        self.active_tracks = 0
        self.track_history.clear()
        self.reset_counts()
        KalmanBoxTracker.count = 0