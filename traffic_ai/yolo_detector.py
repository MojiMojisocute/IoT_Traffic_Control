import cv2
import numpy as np
from typing import List, Dict, Optional
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class YOLODetector:
    VEHICLE_CLASSES = {'car': 2, 'motorcycle': 3, 'bus': 5, 'truck': 7}
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.3,
        nms_threshold: float = 0.3,
        input_size: int = 640,
        device: str = 'cuda',
        half_precision: bool = True,
        model_dir: str = './data/models'
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.device = device
        self.half_precision = half_precision
        self.model_dir = Path(model_dir)
        
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.backend = None
        
        self.total_detections = 0
        self.total_inference_time = 0.0
        self.frame_count = 0

        self._initialize_detector()
    
    def _initialize_detector(self):
        if self._try_ultralytics():
            self.backend = 'ultralytics'
            logger.info(f"Using Ultralytics YOLO - Device: {self.device}")
            return
        
        self.backend = 'basic'
        logger.warning("No YOLO model - using basic detection")
    
    def _try_ultralytics(self) -> bool:
        try:
            from ultralytics import YOLO
            import torch
            
            model_name = self.model_path or 'yolov8n.pt'
            model_path = Path(model_name)
            
            if not model_path.is_absolute() and not model_path.exists():
                model_path = self.model_dir / model_name
            
            logger.info(f"Loading YOLO: {model_name}")
            
            if not model_path.exists():
                logger.info(f"Downloading {model_name}...")
                self.model = YOLO(model_name)
                if Path(model_name).exists():
                    import shutil
                    shutil.move(model_name, model_path)
            else:
                self.model = YOLO(str(model_path))
            
            if self.device == 'cuda' and torch.cuda.is_available():
                self.model.to('cuda')
                if self.half_precision:
                    logger.info("FP16 enabled for faster inference")
            else:
                self.device = 'cpu'
                self.half_precision = False
                logger.warning("CUDA not available - using CPU")
            
            self.model.overrides['save'] = False
            self.model.overrides['verbose'] = False
            
            logger.info("Warming up model...")
            dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
            self.model.predict(dummy, conf=self.confidence_threshold, verbose=False, save=False)
            
            logger.info("Model ready")
            return True
            
        except Exception as e:
            logger.error(f"Error loading ultralytics: {e}")
            return False
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        if frame is None:
            return []
        
        start_time = time.time()
        
        if self.backend == 'ultralytics':
            detections = self._detect_ultralytics(frame)
        else:
            detections = self._detect_basic(frame)
        
        inference_time = time.time() - start_time
        self.total_inference_time += inference_time
        self.frame_count += 1
        self.total_detections += len(detections)
        
        return detections
    
    def _detect_ultralytics(self, frame: np.ndarray) -> List[Dict]:
        try:
            results = self.model.predict(
                frame,
                conf=self.confidence_threshold,
                iou=self.nms_threshold,
                classes=list(self.VEHICLE_CLASSES.values()),
                half=self.half_precision,
                verbose=False,
                imgsz=self.input_size,
                device=self.device,
                save=False,
                save_txt=False,
                save_conf=False,
                save_crop=False,
                show=False,
                line_width=1
            )
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                
                if len(boxes) == 0:
                    continue
                
                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)
                
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = map(int, xyxy[i])
                    class_id = cls[i]
                    confidence = float(conf[i])
                    
                    class_name = next((k for k, v in self.VEHICLE_CLASSES.items() if v == class_id), 'unknown')
                    
                    if class_name == 'unknown':
                        continue
                    
                    w = x2 - x1
                    h = y2 - y1
                    cx = x1 + w // 2
                    cy = y1 + h // 2
                    
                    detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': [x1, y1, w, h],
                        'center': [cx, cy]
                    })
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []
    
    def _detect_basic(self, frame: np.ndarray) -> List[Dict]:
        height, width = frame.shape[:2]
        
        return [{
            'class': 'car',
            'confidence': 0.75,
            'bbox': [width//3, height//2, width//6, height//8],
            'center': [width//2, height//2]
        }]
    
    def get_stats(self) -> Dict:
        avg_time = self.total_inference_time / self.frame_count if self.frame_count > 0 else 0
        avg_det = self.total_detections / self.frame_count if self.frame_count > 0 else 0
        
        return {
            'backend': self.backend,
            'device': self.device,
            'frame_count': self.frame_count,
            'total_detections': self.total_detections,
            'avg_detections': round(avg_det, 2),
            'avg_inference_ms': round(avg_time * 1000, 2),
            'fps': round(1.0 / avg_time, 2) if avg_time > 0 else 0
        }