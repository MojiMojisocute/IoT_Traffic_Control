import cv2
import json
from pathlib import Path


class PolygonSelector:
    def __init__(self, video_path, config_path='data/config/line_config.json'):
        self.video_path = video_path
        self.config_path = config_path
        self.cap = cv2.VideoCapture(video_path)
        
        self.points = []
        self.current_polygon = None
        self.polygons = {}
        
        self.frame = None
        self.frame_display = None
        
        print("=" * 60)
        print("POLYGON SELECTOR - Draw lane boundaries")
        print("=" * 60)
        print("\nInstructions:")
        print("  - Click multiple points to draw a polygon")
        print("  - Double-click (or press 'e') to end polygon and close it")
        print("  - Press 'n' to name the polygon (e.g., 'lane1', 'lane2')")
        print("  - Press 's' to save and exit")
        print("  - Press 'c' to clear current polygon")
        print("  - Press 'd' to delete last polygon")
        print("  - Press 'q' to quit without saving\n")
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            
            self.frame_display = self.frame.copy()
            self._draw_all_polygons(self.frame_display)
            
            if len(self.points) >= 1:
                for pt in self.points:
                    cv2.circle(self.frame_display, pt, 5, (0, 255, 255), -1)
                
                if len(self.points) >= 2:
                    for i in range(len(self.points) - 1):
                        cv2.line(self.frame_display, self.points[i], self.points[i+1], (0, 0, 255), 2)
                
                cv2.putText(self.frame_display, f"Points: {len(self.points)} (Press 'e' to finish)", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._finish_polygon()
    
    def _finish_polygon(self):
        if len(self.points) < 3:
            print("⚠ Need at least 3 points!")
            return
        
        self.current_polygon = self.points.copy()
        self.points = []
        
        self.frame_display = self.frame.copy()
        self._draw_all_polygons(self.frame_display)
        
        cv2.putText(self.frame_display, "Press 'n' to name this polygon", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    def _draw_all_polygons(self, frame):
        for name, polygon in self.polygons.items():
            pts = np.array(polygon, np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            
            cx = sum(p[0] for p in polygon) // len(polygon)
            cy = sum(p[1] for p in polygon) // len(polygon)
            cv2.putText(frame, name, (cx - 20, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if self.current_polygon:
            pts = np.array(self.current_polygon, np.int32)
            cv2.polylines(frame, [pts], True, (0, 0, 255), 3)
            for pt in self.current_polygon:
                cv2.circle(frame, pt, 5, (0, 255, 255), -1)
    
    def run(self):
        ret, self.frame = self.cap.read()
        
        if not ret:
            print("Error: Cannot read video")
            return False
        
        self.frame_display = self.frame.copy()
        
        cv2.namedWindow('Polygon Selector')
        cv2.setMouseCallback('Polygon Selector', self.mouse_callback)
        
        print("Click on video to place points...\n")
        
        while True:
            cv2.imshow('Polygon Selector', self.frame_display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n✗ Cancelled without saving")
                cv2.destroyAllWindows()
                return False
            
            elif key == ord('e'):
                self._finish_polygon()
            
            elif key == ord('n') and self.current_polygon:
                print("Enter name for this polygon (e.g., 'lane1', 'lane2'): ", end='')
                polygon_name = input().strip()
                
                if polygon_name:
                    self.polygons[polygon_name] = self.current_polygon
                    self.current_polygon = None
                    
                    self.frame_display = self.frame.copy()
                    self._draw_all_polygons(self.frame_display)
                    
                    print(f"✓ Added polygon: {polygon_name}")
                    print(f"  Points: {len(self.polygons[polygon_name])}")
                    print(f"  Total polygons: {len(self.polygons)}")
            
            elif key == ord('c'):
                if self.current_polygon or self.points:
                    self.current_polygon = None
                    self.points = []
                    self.frame_display = self.frame.copy()
                    self._draw_all_polygons(self.frame_display)
                    print("Cleared current polygon")
            
            elif key == ord('d'):
                if self.polygons:
                    last_polygon = list(self.polygons.keys())[-1]
                    del self.polygons[last_polygon]
                    self.frame_display = self.frame.copy()
                    self._draw_all_polygons(self.frame_display)
                    print(f"Deleted polygon: {last_polygon}")
            
            elif key == ord('s') and self.polygons:
                self._save_polygons()
                cv2.destroyAllWindows()
                return True
            
            elif key == ord('s') and not self.polygons:
                print("Error: No polygons defined yet!")
        
        cv2.destroyAllWindows()
        return False
    
    def _save_polygons(self):
        Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.polygons, f, indent=2)
        
        print(f"\n✓ Saved {len(self.polygons)} polygons to {self.config_path}")
        print("\nPolygon Configuration:")
        for name, polygon in self.polygons.items():
            print(f"  {name}: {len(polygon)} points")
            for i, pt in enumerate(polygon):
                print(f"    {i+1}. {pt}")


def load_line_config(config_path='data/config/line_config.json'):
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    return {}


import numpy as np

if __name__ == "__main__":
    VIDEO_PATH = "data/testing/test3.mp4"
    CONFIG_PATH = "data/config/line_config.json"
    
    selector = PolygonSelector(VIDEO_PATH, CONFIG_PATH)
    
    if selector.run():
        print("\n✓ Polygon selection completed!")
    else:
        print("\n✗ Polygon selection cancelled")