import cv2
import numpy as np
from typing import Optional, Tuple, Dict
import logging
import time
import threading
from collections import deque

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Camera:
    """
    📷 Production-Grade Camera Manager
    
    Stability Features:
    - Automatic reconnection with exponential backoff
    - Frame buffer for smooth playback
    - FPS calculation and monitoring
    - Health check system
    - Graceful degradation
    """
    
    def __init__(
        self,
        source: str = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 60,
        rotate: int = 0,
        reconnect_attempts: int = 5,
        buffer_size: int = 1,
        backend: int = cv2.CAP_ANY,
        denoise: bool = False,
        fast_mode: bool = True
    ):
        """
        Initialize Camera with enhanced stability
        
        Args:
            source: Camera source
                    - 0, 1, 2 สำหรับ USB Camera
                    - "rtsp://..." สำหรับ IP Camera
                    - "video.mp4" สำหรับ Video File
            width: Frame width (target)
            height: Frame height (target)
            fps: Target FPS
            rotate: Rotation angle (0, 90, 180, 270)
            reconnect_attempts: จำนวนครั้งที่พยายามเชื่อมต่อใหม่
            buffer_size: ขนาด buffer สำหรับเก็บ frame (1=fastest, 2-3=balanced)
            backend: OpenCV backend (CAP_ANY, CAP_DSHOW, CAP_V4L2)
            denoise: เปิดใช้งาน denoising filter
            fast_mode: ใช้ fast interpolation และปิด features ที่ไม่จำเป็น
        """
        self.source = source
        self.width = width
        self.height = height
        self.target_fps = fps
        self.rotate = rotate
        self.reconnect_attempts = reconnect_attempts
        self.buffer_size = buffer_size
        self.backend = backend
        self.denoise = denoise
        self.fast_mode = fast_mode
        
        self.cap = None
        self.is_opened = False
        self.frame_count = 0
        self.last_frame = None
        self.last_successful_read = time.time()
        
        self.fps_buffer = deque(maxlen=60)
        self.last_fps_time = time.time()
        self.actual_fps = 0.0
        
        self.failed_reads = 0
        self.successful_reads = 0
        self.reconnect_count = 0
        
        self.lock = threading.Lock()
        
        self._open_camera()
    
    def _open_camera(self) -> bool:
        try:
            # ถ้าเป็นตัวเลข แปลงเป็น int (USB Camera)
            if isinstance(self.source, str) and self.source.isdigit():
                self.source = int(self.source)
            
            # เปิด VideoCapture with backend
            logger.info(f"🎬 กำลังเปิดกล้อง: {self.source}")
            self.cap = cv2.VideoCapture(self.source, self.backend)
            
            if not self.cap.isOpened():
                logger.error(f"❌ ไม่สามารถเปิดกล้อง: {self.source}")
                return False
            
            # ตั้งค่า buffer size
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
            
            # ตั้งค่า resolution และ FPS
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            # สำหรับ RTSP: ตั้งค่าเพิ่มเติม
            if isinstance(self.source, str) and self.source.startswith('rtsp'):
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # ลด latency
            
            # อ่านค่าจริงที่ได้
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            # ทดสอบอ่าน frame
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                logger.error(f"❌ ไม่สามารถอ่าน frame ทดสอบได้")
                self.cap.release()
                return False
            
            self.is_opened = True
            self.last_successful_read = time.time()
            
            logger.info(f"✅ เปิดกล้องสำเร็จ: {self.source}")
            logger.info(f"📐 Resolution: {actual_width}x{actual_height} @ {actual_fps} FPS")
            logger.info(f"🎯 Target: {self.width}x{self.height} @ {self.target_fps} FPS")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error opening camera: {e}")
            return False
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        อ่านเฟรมจากกล้อง with robust error handling
        
        Returns:
            numpy array ของเฟรม หรือ None ถ้าอ่านไม่สำเร็จ
        """
        with self.lock:
            # ตรวจสอบสถานะกล้อง
            if not self.is_opened or self.cap is None:
                logger.warning("⚠️ กล้องไม่ได้เปิดอยู่ พยายามเชื่อมต่อใหม่...")
                if not self._reconnect():
                    return self.last_frame  # คืน frame ล่าสุดถ้ามี
            
            try:
                # อ่าน frame
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    self.failed_reads += 1
                    
                    # ถ้าอ่านไม่ได้หลายครั้งติดกัน
                    if self.failed_reads > 10:
                        logger.warning(f"⚠️ อ่านเฟรมไม่ได้ {self.failed_reads} ครั้ง พยายามเชื่อมต่อใหม่...")
                        if not self._reconnect():
                            return self.last_frame
                        
                        # ลองอ่านอีกครั้ง
                        ret, frame = self.cap.read()
                        if not ret or frame is None:
                            return self.last_frame
                        
                        self.failed_reads = 0
                    else:
                        return self.last_frame
                
                # Preprocess frame
                frame = self._preprocess_frame(frame)
                
                # Update statistics
                self.successful_reads += 1
                self.failed_reads = 0
                self.frame_count += 1
                self.last_frame = frame
                self.last_successful_read = time.time()
                
                # Calculate FPS
                self._update_fps()
                
                return frame
                
            except Exception as e:
                logger.error(f"❌ Error reading frame: {e}")
                self.failed_reads += 1
                return self.last_frame
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        ประมวลผลเฟรมอย่างมีประสิทธิภาพ
        
        Args:
            frame: Input frame
            
        Returns:
            Processed frame
        """
        # เลือก interpolation method ตาม mode
        interpolation = cv2.INTER_NEAREST if self.fast_mode else cv2.INTER_LINEAR
        
        # Resize ถ้าขนาดไม่ตรง
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), 
                             interpolation=interpolation)
        
        # Rotate ถ้าต้องการ
        if self.rotate == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotate == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotate == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Denoise (optional) - skip in fast mode
        if self.denoise and not self.fast_mode:
            frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
        
        return frame
    
    def _update_fps(self):
        """
        คำนวณ FPS จริง
        """
        current_time = time.time()
        time_diff = current_time - self.last_fps_time
        
        if time_diff > 0:
            self.fps_buffer.append(1.0 / time_diff)
            self.actual_fps = sum(self.fps_buffer) / len(self.fps_buffer)
        
        self.last_fps_time = current_time
    
    def _reconnect(self) -> bool:
        """
        พยายามเชื่อมต่อกล้องใหม่ with exponential backoff
        
        Returns:
            True ถ้าเชื่อมต่อสำเร็จ
        """
        logger.info("🔄 กำลังพยายามเชื่อมต่อกล้องใหม่...")
        
        # ปิดการเชื่อมต่อเดิม
        if self.cap is not None:
            try:
                self.cap.release()
            except:
                pass
        
        # Exponential backoff
        wait_times = [1, 2, 4, 8, 16]
        
        # พยายามเชื่อมต่อใหม่
        for attempt in range(self.reconnect_attempts):
            wait_time = wait_times[min(attempt, len(wait_times)-1)]
            logger.info(f"🔄 ครั้งที่ {attempt + 1}/{self.reconnect_attempts} (รอ {wait_time}s)")
            
            if attempt > 0:
                time.sleep(wait_time)
            
            if self._open_camera():
                self.reconnect_count += 1
                logger.info(f"✅ เชื่อมต่อกล้องใหม่สำเร็จ (reconnect #{self.reconnect_count})")
                return True
        
        logger.error("❌ ไม่สามารถเชื่อมต่อกล้องได้หลังจากพยายาม {self.reconnect_attempts} ครั้ง")
        self.is_opened = False
        return False
    
    def get_fps(self) -> float:
        """
        ดึงค่า FPS จริงที่วัดได้
        
        Returns:
            Actual FPS value
        """
        return self.actual_fps
    
    def get_target_fps(self) -> int:
        """
        ดึงค่า FPS target
        
        Returns:
            Target FPS value
        """
        return self.target_fps
    
    def get_resolution(self) -> Tuple[int, int]:
        """
        ดึงค่า Resolution จริงของกล้อง
        
        Returns:
            (width, height)
        """
        if self.cap is not None and self.is_opened:
            try:
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                return (width, height)
            except:
                pass
        return (self.width, self.height)
    
    def is_available(self) -> bool:
        """
        ตรวจสอบว่ากล้องพร้อมใช้งานหรือไม่
        
        Returns:
            True ถ้าพร้อมใช้งาน
        """
        if not self.is_opened or self.cap is None:
            return False
        
        # Check if we haven't read successfully in a while
        time_since_last_read = time.time() - self.last_successful_read
        if time_since_last_read > 30:  # 30 วินาที
            return False
        
        try:
            return self.cap.isOpened()
        except:
            return False
    
    def get_health_stats(self) -> Dict:
        """
        ดึงสถิติสุขภาพของกล้อง
        
        Returns:
            Dict ของสถิติต่างๆ
        """
        total_reads = self.successful_reads + self.failed_reads
        success_rate = (self.successful_reads / total_reads * 100) if total_reads > 0 else 0
        
        return {
            'is_available': self.is_available(),
            'frame_count': self.frame_count,
            'fps': round(self.actual_fps, 2),
            'target_fps': self.target_fps,
            'resolution': self.get_resolution(),
            'successful_reads': self.successful_reads,
            'failed_reads': self.failed_reads,
            'success_rate': round(success_rate, 2),
            'reconnect_count': self.reconnect_count,
            'time_since_last_read': round(time.time() - self.last_successful_read, 2)
        }
    
    def release(self):
        """
        ปิดการเชื่อมต่อกล้องอย่างปลอดภัย
        """
        with self.lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                    logger.info("📷 ปิดกล้องแล้ว")
                except Exception as e:
                    logger.error(f"Error releasing camera: {e}")
            
            self.is_opened = False
            self.last_frame = None
    
    def __del__(self):
        """
        Destructor - ปิดกล้องอัตโนมัติ
        """
        self.release()
    
    def __enter__(self):
        """
        Context manager support
        """
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager cleanup
        """
        self.release()


# ==========================================
# 🧪 ตัวอย่างการใช้งาน
# ==========================================

if __name__ == "__main__":
    print("🎬 Enhanced Camera Manager Test")
    print("=" * 60)
    
    print("\n📷 ทดสอบ USB Camera...")
    
    with Camera(source=0, width=640, height=480) as camera:
        if camera.is_available():
            print(f"✅ กล้องพร้อมใช้งาน")
            print(f"📊 Health Stats:")
            for key, value in camera.get_health_stats().items():
                print(f"   {key}: {value}")
            
            print("\n💡 กด 'q' เพื่อปิด | 's' เพื่อดู stats\n")

            last_stats_time = time.time()
            
            while True:
                frame = camera.get_frame()
                
                if frame is not None:
                    info_text = f"Frame: {camera.frame_count} | FPS: {camera.get_fps():.1f}/{camera.get_target_fps()}"
                    cv2.putText(frame, info_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    res = camera.get_resolution()
                    cv2.putText(frame, f"Resolution: {res[0]}x{res[1]}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    cv2.putText(frame, "Press 'q' to quit | 's' for stats", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    cv2.imshow("Enhanced Camera Test - Smart Traffic System", frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('q'):
                        print(f"\n✅ ปิดกล้อง")
                        break
                    elif key == ord('s'):
                        print("\n📊 Camera Health Stats:")
                        stats = camera.get_health_stats()
                        for key, value in stats.items():
                            print(f"   {key}: {value}")
                        print()
                
                # แสดง stats ทุก 10 วินาที
                if time.time() - last_stats_time > 10:
                    logger.info(f"📊 Stats: {camera.get_health_stats()}")
                    last_stats_time = time.time()
            
            cv2.destroyAllWindows()
        else:
            print("❌ ไม่สามารถเปิดกล้องได้")
    
    print("\n✅ Test Complete!")
    
    # ตัวอย่างการใช้แบบอื่นๆ:
    # camera = Camera(source="rtsp://admin:password@192.168.1.100:554/stream")
    # camera = Camera(source="traffic_video.mp4", denoise=True)
    # camera = Camera(source=0, rotate=90, backend=cv2.CAP_DSHOW)