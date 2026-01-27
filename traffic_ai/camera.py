import cv2
import numpy as np
from typing import Optional, Tuple, Dict
import logging
import time
import threading
from collections import deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Camera:
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
        use_cuda: bool = True
    ):
        self.source = source
        self.width = width
        self.height = height
        self.target_fps = fps
        self.rotate = rotate
        self.reconnect_attempts = reconnect_attempts
        self.buffer_size = buffer_size
        self.backend = backend
        self.use_cuda = use_cuda
        
        self.cap = None
        self.is_opened = False
        self.frame_count = 0
        self.last_frame = None
        self.last_successful_read = time.time()
        
        self.fps_buffer = deque(maxlen=30)
        self.last_fps_time = time.time()
        self.actual_fps = 0.0
        
        self.failed_reads = 0
        self.successful_reads = 0
        self.reconnect_count = 0
        
        self.lock = threading.Lock()
        
        self.cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0 if self.use_cuda else False
        if self.cuda_available:
            logger.info(f"CUDA GPU available: {cv2.cuda.getCudaEnabledDeviceCount()} device(s)")
        
        self._open_camera()
    
    def _open_camera(self) -> bool:
        try:
            if isinstance(self.source, str) and self.source.isdigit():
                self.source = int(self.source)
            
            logger.info(f"Opening camera: {self.source}")
            self.cap = cv2.VideoCapture(self.source, self.backend)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera: {self.source}")
                return False
            
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            if isinstance(self.source, str) and self.source.startswith('rtsp'):
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                logger.error(f"Failed to read test frame")
                self.cap.release()
                return False
            
            self.is_opened = True
            self.last_successful_read = time.time()
            
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"Camera opened: {actual_width}x{actual_height}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error opening camera: {e}")
            return False
    
    def get_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            if not self.is_opened or self.cap is None:
                if not self._reconnect():
                    return self.last_frame
            
            try:
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    self.failed_reads += 1
                    
                    if self.failed_reads > 10 and not self._reconnect():
                        return self.last_frame
                    
                    return self.last_frame
                
                frame = self._preprocess_frame(frame)
                
                self.successful_reads += 1
                self.failed_reads = 0
                self.frame_count += 1
                self.last_frame = frame
                self.last_successful_read = time.time()
                
                self._update_fps()
                
                return frame
                
            except Exception as e:
                logger.error(f"Error reading frame: {e}")
                self.failed_reads += 1
                return self.last_frame
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            if self.cuda_available:
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                gpu_resized = cv2.cuda.resize(gpu_frame, (self.width, self.height))
                frame = gpu_resized.download()
            else:
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        
        if self.rotate == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotate == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotate == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        return frame
    
    def _update_fps(self):
        current_time = time.time()
        time_diff = current_time - self.last_fps_time
        
        if time_diff > 0:
            self.fps_buffer.append(1.0 / time_diff)
            self.actual_fps = sum(self.fps_buffer) / len(self.fps_buffer)
        
        self.last_fps_time = current_time
    
    def _reconnect(self) -> bool:
        if self.cap is not None:
            try:
                self.cap.release()
            except:
                pass
        
        wait_times = [1, 2, 4, 8, 16]
        
        for attempt in range(self.reconnect_attempts):
            wait_time = wait_times[min(attempt, len(wait_times)-1)]
            
            if attempt > 0:
                time.sleep(wait_time)
            
            if self._open_camera():
                self.reconnect_count += 1
                return True
        
        self.is_opened = False
        return False
    
    def get_fps(self) -> float:
        return self.actual_fps
    
    def get_resolution(self) -> Tuple[int, int]:
        if self.cap is not None and self.is_opened:
            try:
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                return (width, height)
            except:
                pass
        return (self.width, self.height)
    
    def is_available(self) -> bool:
        if not self.is_opened or self.cap is None:
            return False
        
        time_since_last_read = time.time() - self.last_successful_read
        if time_since_last_read > 30:
            return False
        
        try:
            return self.cap.isOpened()
        except:
            return False
    
    def get_health_stats(self) -> Dict:
        total_reads = self.successful_reads + self.failed_reads
        success_rate = (self.successful_reads / total_reads * 100) if total_reads > 0 else 0
        
        return {
            'is_available': self.is_available(),
            'frame_count': self.frame_count,
            'fps': round(self.actual_fps, 2),
            'resolution': self.get_resolution(),
            'cuda_enabled': self.cuda_available,
            'success_rate': round(success_rate, 2)
        }
    
    def release(self):
        with self.lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except:
                    pass
            
            self.is_opened = False
            self.last_frame = None
    
    def __del__(self):
        self.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()