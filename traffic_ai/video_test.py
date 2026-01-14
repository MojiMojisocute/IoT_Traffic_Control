"""
🎬 Video Test Script for Smart Traffic AI
ทดสอบ Camera + YOLO กับ video file

Usage:
    python video_test.py                    # ให้เลือก video จาก GUI
    python video_test.py path/to/video.mp4  # ระบุ path โดยตรง
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from camera import Camera
from yolo_detector import YOLODetector
import cv2
import time
from pathlib import Path


def select_video_file():
    """
    เลือก video file ด้วย GUI
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw()
        
        print("🎬 กรุณาเลือก video file...")
        video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
                ("All files", "*.*")
            ]
        )
        
        return video_path if video_path else None
        
    except ImportError:
        print("⚠️ ไม่มี tkinter - กรุณาระบุ path ด้วย command line")
        return None


def test_video(video_path, show_preview=True, save_output=False):
    """
    ทดสอบกับ video file
    
    Args:
        video_path: path to video file
        show_preview: แสดง preview window หรือไม่
        save_output: บันทึก output video หรือไม่
    """
    print("\n" + "="*60)
    print("🎬 Video Test - Smart Traffic AI")
    print("="*60)
    
    # ตรวจสอบไฟล์
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"❌ ไม่พบไฟล์: {video_path}")
        return False
    
    print(f"\n✅ พบ video: {video_path.name}")
    print(f"📁 Path: {video_path.absolute()}")
    
    # เปิด video
    print("\n🔄 กำลังเปิด video...")
    camera = Camera(
        source=str(video_path),
        width=1280,  # ปรับตามต้องการ
        height=720
    )
    
    if not camera.is_available():
        print("❌ ไม่สามารถเปิด video ได้")
        return False
    
    print("✅ เปิด video สำเร็จ")
    
    # แสดงข้อมูล video
    res = camera.get_resolution()
    fps = camera.get_target_fps()
    print(f"📐 Resolution: {res[0]}x{res[1]}")
    print(f"⚡ FPS: {fps}")
    
    # สร้าง detector
    print("\n🧠 กำลังสร้าง YOLO detector...")
    detector = YOLODetector(
        confidence_threshold=0.5,
        nms_threshold=0.4
    )
    print(f"✅ Backend: {detector.backend}")
    
    # Setup output video (optional)
    video_writer = None
    if save_output:
        output_path = video_path.parent / f"{video_path.stem}_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            res
        )
        print(f"💾 Output จะถูกบันทึกที่: {output_path}")
    
    # Process video
    print("\n🎬 เริ่มประมวลผล video...")
    print("💡 Controls:")
    print("   'q' = Quit")
    print("   'p' = Pause/Resume")
    print("   's' = Save screenshot")
    print("   'i' = Show info")
    print("   'SPACE' = Pause/Resume")
    
    frame_count = 0
    total_detections = 0
    start_time = time.time()
    paused = False
    
    try:
        while True:
            if not paused:
                # อ่าน frame
                frame = camera.get_frame()
                
                if frame is None:
                    print("\n✅ จบ video")
                    break
                
                # Detect
                detections = detector.detect(frame)
                total_detections += len(detections)
                frame_count += 1
                
                # Draw detections
                frame_with_boxes = detector.draw_detections(frame, detections)
                
                # เพิ่มข้อมูลบนภาพ
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                
                info_lines = [
                    f"Frame: {frame_count} | FPS: {current_fps:.1f}",
                    f"Detections: {len(detections)} | Total: {total_detections}",
                    f"Backend: {detector.backend}",
                    f"Time: {elapsed:.1f}s"
                ]
                
                y_offset = 30
                for line in info_lines:
                    cv2.putText(
                        frame_with_boxes,
                        line,
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )
                    y_offset += 30
                
                # Controls hint
                cv2.putText(
                    frame_with_boxes,
                    "Press 'q' to quit | 'p' to pause",
                    (10, frame_with_boxes.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
                
                # บันทึก output
                if video_writer is not None:
                    video_writer.write(frame_with_boxes)
                
                # แสดงผล
                if show_preview:
                    cv2.imshow("Traffic AI - Video Test", frame_with_boxes)
                
                # Progress
                if frame_count % 30 == 0:
                    print(f"📊 Frame {frame_count} | Detections: {len(detections)} | FPS: {current_fps:.1f}")
            
            # Keyboard control
            if show_preview:
                key = cv2.waitKey(1 if not paused else 0) & 0xFF
                
                if key == ord('q'):
                    print("\n👋 หยุดประมวลผล...")
                    break
                elif key == ord('p') or key == ord(' '):
                    paused = not paused
                    print(f"⏸️  {'Paused' if paused else 'Resumed'}")
                elif key == ord('s'):
                    # Save screenshot
                    screenshot_path = video_path.parent / f"screenshot_{frame_count}.jpg"
                    cv2.imwrite(str(screenshot_path), frame_with_boxes)
                    print(f"📸 Saved: {screenshot_path}")
                elif key == ord('i'):
                    # Show detailed info
                    print("\n" + "="*60)
                    print("📊 Current Stats:")
                    print("="*60)
                    print(f"Camera: {camera.get_health_stats()}")
                    print(f"Detector: {detector.get_stats()}")
                    print("="*60 + "\n")
            else:
                # ถ้าไม่แสดง preview ให้หยุดด้วย Ctrl+C
                time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n⚠️ หยุดโดย user")
    
    finally:
        # Cleanup
        camera.release()
        if video_writer is not None:
            video_writer.release()
        if show_preview:
            cv2.destroyAllWindows()
        
        # แสดงสถิติสุดท้าย
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("📊 Final Statistics")
        print("="*60)
        
        print(f"\n📹 Video Info:")
        print(f"   File: {video_path.name}")
        print(f"   Processed Frames: {frame_count}")
        print(f"   Duration: {elapsed_time:.2f}s")
        print(f"   Average FPS: {frame_count/elapsed_time:.2f}")
        
        print(f"\n🚗 Detection Stats:")
        print(f"   Total Detections: {total_detections}")
        print(f"   Avg Detections/Frame: {total_detections/frame_count:.2f}")
        
        print(f"\n🧠 Detector Info:")
        detector_stats = detector.get_stats()
        for key, value in detector_stats.items():
            print(f"   {key}: {value}")
        
        print(f"\n📷 Camera Health:")
        camera_stats = camera.get_health_stats()
        for key, value in camera_stats.items():
            print(f"   {key}: {value}")
        
        if save_output:
            print(f"\n💾 Output saved to: {output_path}")
        
        print("\n" + "="*60)
        print("✅ Test Completed!")
        print("="*60)
    
    return True


def main():
    """
    Main function
    """
    print("\n🎬 Smart Traffic AI - Video Test Tool")
    print("="*60)
    
    # Get video path
    video_path = None
    
    if len(sys.argv) > 1:
        # ระบุ path ผ่าน command line
        video_path = sys.argv[1]
        print(f"📁 Using video from command line: {video_path}")
    else:
        # เลือกด้วย GUI
        video_path = select_video_file()
        
        if not video_path:
            print("\n❌ ไม่ได้เลือก video file")
            print("\nUsage:")
            print("   python video_test.py path/to/video.mp4")
            print("   หรือรันแบบไม่มี argument เพื่อเลือกด้วย GUI")
            return
    
    # ถาม options
    print("\n⚙️  Options:")
    print("1. Show preview window? (y/n) [y]: ", end='')
    show_preview = input().strip().lower() != 'n'
    
    print("2. Save output video? (y/n) [n]: ", end='')
    save_output = input().strip().lower() == 'y'
    
    # Run test
    test_video(video_path, show_preview, save_output)


if __name__ == "__main__":
    main()