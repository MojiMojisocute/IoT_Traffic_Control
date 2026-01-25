import cv2
import numpy as np
import time
import sys
import serial
import json
import os
import shutil
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
os.environ['YOLOv8_VERBOSE'] = '0'

try:
    from camera import Camera
    from yolo_detector import YOLODetector
    from vehicle_tracker import VehicleTrackerCounter
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def cleanup_runs_folder():
    if os.path.exists('runs'):
        try:
            shutil.rmtree('runs')
            print("✓ Deleted runs/ folder")
        except Exception as e:
            print(f"⚠ Could not delete runs/: {e}")


def draw_tracks(frame, tracks):
    color = (0, 255, 0)
    
    for track in tracks:
        x, y, w, h = track['bbox']
        track_id = track['track_id']
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        label = f"ID:{track_id}"
        cv2.putText(frame, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cx, cy = track['center']
        cv2.circle(frame, (cx, cy), 4, color, -1)


def draw_stats(frame, tracker, detector, counts, fps):
    y = 30
    
    cv2.putText(frame, f"FPS: {fps:.1f} | Frame: {tracker.frame_count}", 
               (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    y += 30
    cv2.putText(frame, f"Active: {tracker.active_tracks} | Total: {tracker.total_tracks}", 
               (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    y += 25
    cv2.putText(frame, f"Device: {detector.device.upper()}", 
               (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    y += 30
    cv2.putText(frame, "Q=Quit | S=Stats | P=Pause", 
               (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def main():
    print("=" * 80)
    print("SMART TRAFFIC MONITORING SYSTEM - SERIAL COMMUNICATION")
    print("=" * 80)
    
    cleanup_runs_folder()
    
    VIDEO_PATH = "data/testing/test3.mp4"
    SERIAL_PORT = "/dev/ttyUSB0"
    SERIAL_BAUDRATE = 115200
    SERIAL_UPDATE_INTERVAL = 1.0
    
    if not Path(VIDEO_PATH).exists():
        print(f"✗ Video not found: {VIDEO_PATH}")
        return
    
    print(f"✓ Video: {VIDEO_PATH}")
    
    print("\n[1/4] Initializing Camera...")
    try:
        camera = Camera(
            source=VIDEO_PATH,
            width=1280,
            height=720,
            fps=30,
            use_cuda=True
        )
        
        if not camera.is_available():
            print("Failed to open camera")
            return
        
        width, height = camera.get_resolution()
        print(f"✓ Camera: {width}x{height} | CUDA: {camera.cuda_available}")
    except Exception as e:
        print(f"✗ Camera error: {e}")
        return
    
    print("\n[2/4] Initializing YOLO...")
    try:
        detector = YOLODetector(
            confidence_threshold=0.3,
            nms_threshold=0.3,
            input_size=640,
            device='cuda',
            half_precision=True
        )
        print(f"✓ YOLO: {detector.backend} | Device: {detector.device}")
    except Exception as e:
        print(f"✗ YOLO error: {e}")
        camera.release()
        return
    
    cleanup_runs_folder()
    
    print("\n[3/4] Initializing Tracker...")
    try:
        tracker = VehicleTrackerCounter(
            frame_width=width,
            frame_height=height,
            max_age=30,
            min_hits=3,
            iou_threshold=0.3
        )
        print(f"✓ Tracker ready")
    except Exception as e:
        print(f"✗ Tracker error: {e}")
        camera.release()
        return
    
    print("\n[4/4] Initializing Serial Connection...")
    serial_port = None
    try:
        serial_port = serial.Serial(
            port=SERIAL_PORT,
            baudrate=SERIAL_BAUDRATE,
            timeout=1
        )
        print(f"✓ Serial: {SERIAL_PORT} at {SERIAL_BAUDRATE} baud")
    except Exception as e:
        print(f"⚠ Serial connection failed: {e}")
        print("  Continuing without serial communication")
    
    print("\n" + "=" * 80)
    print("System Ready!")
    print("=" * 80)
    
    fps_buffer = []
    last_time = time.time()
    last_serial_time = time.time()
    paused = False
    frame = None
    
    try:
        while True:
            if not paused:
                frame = camera.get_frame()
                
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                detections = detector.detect(frame)
                tracks, counts = tracker.update(detections)
                
                cleanup_runs_folder()
                
                current_time = time.time()
                fps = 1.0 / (current_time - last_time) if (current_time - last_time) > 0 else 0
                last_time = current_time
                
                fps_buffer.append(fps)
                if len(fps_buffer) > 30:
                    fps_buffer.pop(0)
                avg_fps = sum(fps_buffer) / len(fps_buffer)
                
                draw_tracks(frame, tracks)
                draw_stats(frame, tracker, detector, counts, avg_fps)
                
                if serial_port and serial_port.is_open:
                    if current_time - last_serial_time >= SERIAL_UPDATE_INTERVAL:
                        active_count = tracker.active_tracks
                        
                        if active_count <= 5:
                            density = "low"
                        elif active_count <= 15:
                            density = "medium"
                        else:
                            density = "high"
                        
                        data = {
                            "count": active_count,
                            "density": density,
                            "timestamp": int(current_time)
                        }
                        
                        try:
                            payload = json.dumps(data) + "\n"
                            serial_port.write(payload.encode('utf-8'))
                        except:
                            pass
                        
                        last_serial_time = current_time
            
            cv2.imshow("Smart Traffic Monitoring - Serial", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                print("\n" + "=" * 60)
                print("SYSTEM STATISTICS")
                print("=" * 60)
                print("\nCamera:")
                for k, v in camera.get_health_stats().items():
                    print(f"   {k}: {v}")
                print("\nDetector:")
                for k, v in detector.get_stats().items():
                    print(f"   {k}: {v}")
                print("\nTracker:")
                for k, v in tracker.get_stats().items():
                    print(f"   {k}: {v}")
                print("=" * 60 + "\n")
            elif key == ord('p'):
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'}")
    
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nCleaning up...")
        
        if serial_port and serial_port.is_open:
            try:
                data = {"status": "offline", "timestamp": int(time.time())}
                payload = json.dumps(data) + "\n"
                serial_port.write(payload.encode('utf-8'))
                time.sleep(0.2)
            except:
                pass
            serial_port.close()
        
        camera.release()
        cv2.destroyAllWindows()
        
        cleanup_runs_folder()
        
        print("\n" + "=" * 80)
        print("FINAL STATISTICS")
        print("=" * 80)
        
        print("\nCamera:")
        for k, v in camera.get_health_stats().items():
            print(f"   {k}: {v}")
        
        print("\nDetector:")
        for k, v in detector.get_stats().items():
            print(f"   {k}: {v}")
        
        print("\nTracker:")
        for k, v in tracker.get_stats().items():
            print(f"   {k}: {v}")
        
        print("\n" + "=" * 80)
        print("System Stopped")
        print("=" * 80)


if __name__ == "__main__":
    main()