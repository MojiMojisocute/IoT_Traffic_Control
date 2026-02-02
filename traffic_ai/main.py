#!/usr/bin/env python3

import cv2
import numpy as np
import time
import sys
import serial
import os
import shutil
import warnings
from pathlib import Path
from collections import deque

warnings.filterwarnings('ignore')
os.environ['YOLO_VERBOSE'] = 'False'

try:
    from camera import Camera
    from vehicle_detector import YOLODetector
    from vehicle_tracker import VehicleTrackerCounter
    from roi_selector import load_line_config
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def cleanup_runs_folder():
    if os.path.exists('runs'):
        try:
            shutil.rmtree('runs')
        except:
            pass


def draw_tracks(frame, tracks, counting_lines=None):
    for track in tracks:
        x, y, w, h = track['bbox']
        track_id = track['track_id']
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"ID:{track_id}", (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cx, cy = track['center']
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
    
    if counting_lines:
        for name, polygon in counting_lines.items():
            pts = np.array(polygon, np.int32)
            cv2.polylines(frame, [pts], True, (255, 0, 0), 2)


def draw_stats(frame, tracker, fps, counting_time, serial_status):
    y = 30
    
    cv2.putText(frame, f"FPS: {fps:.1f}", 
               (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    y += 30
    cv2.putText(frame, f"Vehicles: {tracker.active_tracks}", 
               (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    y += 30
    cv2.putText(frame, f"Time: {counting_time}s", 
               (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    y += 30
    color = (0, 255, 0) if serial_status else (0, 0, 255)
    status = "OK" if serial_status else "ERROR"
    cv2.putText(frame, f"Serial: {status}", 
               (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def main():
    print("=" * 80)
    print(" SMART TRAFFIC AI SYSTEM")
    print("=" * 80 + "\n")
    
    cleanup_runs_folder()
    
    VIDEO_PATH = "data/testing/test3.mp4"
    SERIAL_PORT = "/dev/ttyUSB0"
    
    if not Path(VIDEO_PATH).exists():
        print(f"ERROR: Video not found: {VIDEO_PATH}")
        return
    
    print("[1/5] Opening serial port...")
    ser = None
    try:
        ser = serial.Serial(SERIAL_PORT, 115200, timeout=1)
        time.sleep(2)
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        print(f"Serial port opened: {SERIAL_PORT}")
    except Exception as e:
        print(f"WARNING: Cannot open serial port: {e}")
        print("Continue without serial communication")
    
    print("\n[2/5] Initializing camera...")
    camera = Camera(source=VIDEO_PATH, width=1280, height=720, fps=60, use_cuda=True)
    if not camera.is_available():
        print("ERROR: Camera initialization failed")
        if ser:
            ser.close()
        return
    print("Camera ready")
    
    print("\n[3/5] Loading YOLO detector...")
    detector = YOLODetector(
        confidence_threshold=0.1, 
        nms_threshold=0.1, 
        device='cuda', 
        half_precision=True
    )
    print("YOLO ready")
    
    print("\n[4/5] Initializing vehicle tracker...")
    counting_lines = load_line_config('data/config/line_config.json')
    tracker = VehicleTrackerCounter(
        frame_width=1280, 
        frame_height=720,
        max_age=30, 
        min_hits=3, 
        iou_threshold=0.3,
        counting_lines=counting_lines
    )
    print("Tracker ready")
    
    print("\n[5/5] Starting main loop...")
    print("\n" + "=" * 80)
    print("SYSTEM RUNNING - Press 'q' to quit")
    print("=" * 80 + "\n")
    
    fps_buffer = deque(maxlen=30)
    last_frame_time = time.time()
    last_serial_send = time.time()
    
    counting_time = -1
    start_time = None
    
    frame_count = 0
    serial_send_count = 0
    serial_ok = False
    
    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            
            detections = detector.detect(frame)
            tracks, _ = tracker.update(detections)
            
            current_time = time.time()
            
            fps = 1.0 / (current_time - last_frame_time) if (current_time - last_frame_time) > 0 else 0
            last_frame_time = current_time
            fps_buffer.append(fps)
            avg_fps = sum(fps_buffer) / len(fps_buffer)
            
            frame_count += 1
            
            if current_time - last_serial_send >= 1.0:
                count = tracker.active_tracks
                
                if count <= 5:
                    density = "low"
                elif count <= 15:
                    density = "medium"
                else:
                    density = "high"
                
                if start_time is None:
                    start_time = current_time
                    counting_time = -1
                else:
                    counting_time = int(current_time - start_time)
                
                if ser and ser.is_open:
                    try:
                        message = f"{count},{counting_time},{density}\n"
                        ser.write(message.encode('utf-8'))
                        ser.flush()
                        serial_send_count += 1
                        serial_ok = True
                        print(f"[TX] count={count}, time={counting_time}, density={density}")
                    except Exception as e:
                        serial_ok = False
                        print(f"[ERROR] Serial send failed: {e}")
                
                last_serial_send = current_time
            
            draw_tracks(frame, tracks, tracker.counting_lines)
            draw_stats(frame, tracker, avg_fps, counting_time, serial_ok)
            
            cv2.imshow("Traffic AI", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    
    finally:
        print("\n" + "=" * 80)
        print("SHUTTING DOWN")
        print("=" * 80)
        
        print(f"\nFrames processed: {frame_count}")
        print(f"Serial messages sent: {serial_send_count}")
        
        if ser and ser.is_open:
            ser.close()
            print("Serial port closed")
        
        camera.release()
        print("Camera released")
        
        cv2.destroyAllWindows()
        cleanup_runs_folder()
        print("Cleanup complete\n")


if __name__ == "__main__":
    main()