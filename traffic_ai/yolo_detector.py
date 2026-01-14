import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YOLODetector:
    VEHICLE_CLASSES = {
        'car': 2,
        'motorcycle': 3,
        'bus': 5,
        'truck': 7
    }
    
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        input_size: int = 640,
        device: str = 'cpu',
        half_precision: bool = False,
        model_dir: str = './data/models'
    ):
        """
        Initialize YOLO Detector
        
        Args:
            model_path: Path to YOLO model weights
                       - None = use default YOLOv8n
                       - "yolov8n.pt" = YOLOv8 nano
                       - "yolov8s.pt" = YOLOv8 small
                       - "yolov5s.pt" = YOLOv5 small
            confidence_threshold: Minimum confidence for detection (0.0-1.0)
            nms_threshold: NMS threshold for overlapping boxes (0.0-1.0)
            input_size: Input size for YOLO (320, 416, 640)
            device: 'cpu' or 'cuda'
            half_precision: Use FP16 for faster inference (requires CUDA)
            model_dir: Directory to save/load models
        """
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
            logger.info("✅ ใช้ Ultralytics YOLO backend")
            return
        
        if self._try_opencv_dnn():
            self.backend = 'opencv_dnn'
            logger.info("✅ ใช้ OpenCV DNN backend")
            return
        
        self.backend = 'basic'
        logger.warning("⚠️ ไม่มี YOLO model - ใช้ basic detection (สำหรับเทสเท่านั้น)")
    
    def _try_ultralytics(self) -> bool:
        try:
            from ultralytics import YOLO
            
            if self.model_path:
                model_name = self.model_path
            else:
                model_name = 'yolov8n.pt'
            
            model_path = Path(model_name)
            if not model_path.is_absolute() and not model_path.exists():
                model_path = self.model_dir / model_name
            
            logger.info(f"🔄 กำลังโหลด YOLO model: {model_name}")
            
            if not model_path.exists():
                logger.info(f"📥 ดาวน์โหลด {model_name} ไปที่ {self.model_dir}...")
                self.model = YOLO(model_name)
                if Path(model_name).exists():
                    import shutil
                    shutil.move(model_name, model_path)
                    logger.info(f"✅ บันทึก model ที่: {model_path}")
            else:
                logger.info(f"✅ ใช้ model จาก: {model_path}")
                self.model = YOLO(str(model_path))
            
            if self.device == 'cuda':
                self.model.to('cuda')
                if self.half_precision:
                    logger.info("⚡ เปิดใช้ Half Precision (FP16)")
            
            logger.info("🔥 Warming up model...")
            dummy_img = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
            self.model.predict(
                dummy_img, 
                conf=self.confidence_threshold,
                iou=self.nms_threshold,
                half=self.half_precision,
                verbose=False
            )
            
            logger.info(f"✅ Model พร้อมใช้งาน")
            return True
            
        except ImportError:
            logger.warning("⚠️ ไม่พบ ultralytics - ติดตั้งด้วย: pip install ultralytics")
            return False
        except Exception as e:
            logger.error(f"❌ Error loading ultralytics: {e}")
            return False
    
    def _try_opencv_dnn(self) -> bool:
        try:
            if not self.model_path:
                logger.warning("⚠️ ไม่ได้ระบุ model_path สำหรับ OpenCV DNN")
                return False
            
            weights_path = Path(self.model_path)
            config_path = weights_path.with_suffix('.cfg')
            
            if not weights_path.exists():
                logger.warning(f"⚠️ ไม่พบ weights file: {weights_path}")
                return False
            
            self.model = cv2.dnn.readNetFromDarknet(str(config_path), str(weights_path))
            
            self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            logger.info(f"✅ โหลด OpenCV DNN model สำเร็จ")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading OpenCV DNN: {e}")
            return False
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect vehicles in frame
        
        Args:
            frame: Input frame (BGR)
        
        Returns:
            List of detections:
            [
                {
                    'class': 'car',
                    'confidence': 0.92,
                    'bbox': [x, y, w, h],
                    'center': [cx, cy]
                },
                ...
            ]
        """
        if frame is None:
            return []
        
        start_time = time.time()
        
        if self.backend == 'ultralytics':
            detections = self._detect_ultralytics(frame)
        elif self.backend == 'opencv_dnn':
            detections = self._detect_opencv_dnn(frame)
        else:
            detections = self._detect_basic(frame)
        
        # Update stats
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
                stream=False,
                imgsz=self.input_size 
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
                    x1, y1, x2, y2 = xyxy[i]
                    confidence = float(conf[i])
                    class_id = cls[i]
                    
                    # Convert to [x, y, w, h]
                    x, y = int(x1), int(y1)
                    w, h = int(x2 - x1), int(y2 - y1)
                                    
                    class_name = self.COCO_CLASSES[class_id] if class_id < len(self.COCO_CLASSES) else 'unknown'
                                        
                    if class_name not in self.VEHICLE_CLASSES:
                        continue
                    
                    detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': [x, y, w, h],
                        'center': [x + w//2, y + h//2]
                    })
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in ultralytics detection: {e}")
            return []
    
    def _detect_opencv_dnn(self, frame: np.ndarray) -> List[Dict]:
        try:
            height, width = frame.shape[:2]
            
            blob = cv2.dnn.blobFromImage(
                frame, 
                1/255.0, 
                (self.input_size, self.input_size),
                swapRB=True, 
                crop=False
            )
            
            self.model.setInput(blob)
            layer_names = self.model.getLayerNames()
            output_layers = [layer_names[i - 1] for i in self.model.getUnconnectedOutLayers()]
            outputs = self.model.forward(output_layers)
            
            detections = []
            boxes = []
            confidences = []
            class_ids = []
            
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > self.confidence_threshold:
                        class_name = self.COCO_CLASSES[class_id] if class_id < len(self.COCO_CLASSES) else 'unknown'
                        
                        if class_name not in self.VEHICLE_CLASSES:
                            continue
                        
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            if len(boxes) > 0:
                indices = cv2.dnn.NMSBoxes(
                    boxes, 
                    confidences, 
                    self.confidence_threshold, 
                    self.nms_threshold
                )
                
                if len(indices) > 0:
                    for i in indices.flatten():
                        box = boxes[i]
                        class_name = self.COCO_CLASSES[class_ids[i]]
                        
                        detections.append({
                            'class': class_name,
                            'confidence': confidences[i],
                            'bbox': box,
                            'center': [box[0] + box[2]//2, box[1] + box[3]//2]
                        })
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in OpenCV DNN detection: {e}")
            return []
    
    def _detect_basic(self, frame: np.ndarray) -> List[Dict]:
        """
        Basic detection using background subtraction
        (สำหรับ testing เท่านั้น - ไม่แม่นยำ)
        """
        # สร้าง dummy detections สำหรับเทส
        # ในการใช้งานจริงต้องมี YOLO model
        
        height, width = frame.shape[:2]
        
        detections = [
            {
                'class': 'car',
                'confidence': 0.75,
                'bbox': [width//3, height//2, width//6, height//8],
                'center': [width//2, height//2]
            }
        ]
        
        return detections
    
    def get_stats(self) -> Dict:
        avg_inference_time = (
            self.total_inference_time / self.frame_count 
            if self.frame_count > 0 else 0
        )
        
        avg_detections = (
            self.total_detections / self.frame_count 
            if self.frame_count > 0 else 0
        )
        
        return {
            'backend': self.backend,
            'frame_count': self.frame_count,
            'total_detections': self.total_detections,
            'avg_detections_per_frame': round(avg_detections, 2),
            'avg_inference_time': round(avg_inference_time * 1000, 2),  # ms
            'fps': round(1.0 / avg_inference_time, 2) if avg_inference_time > 0 else 0
        }
    
    def draw_detections(
        self, 
        frame: np.ndarray, 
        detections: List[Dict],
        show_confidence: bool = True
    ) -> np.ndarray:
        """
        วาด bounding boxes บนภาพ
        
        Args:
            frame: Input frame
            detections: List of detections
            show_confidence: แสดง confidence score หรือไม่
        
        Returns:
            Frame with drawn boxes
        """
        frame_copy = frame.copy()
        
        colors = {
            'car': (0, 255, 0),      # เขียว
            'bus': (255, 0, 0),      # น้ำเงิน
            'truck': (0, 0, 255),    # แดง
            'motorcycle': (255, 255, 0)  # ฟ้า
        }
        
        for det in detections:
            x, y, w, h = det['bbox']
            cls = det['class']
            conf = det['confidence']
            
            # เลือกสี
            color = colors.get(cls, (255, 255, 255))
            
            # วาด rectangle
            cv2.rectangle(frame_copy, (x, y), (x+w, y+h), color, 2)
            
            # วาด label
            label = f"{cls}"
            if show_confidence:
                label += f" {conf:.2f}"
            
            # Background สำหรับ text
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                frame_copy,
                (x, y - text_height - 10),
                (x + text_width, y),
                color,
                -1
            )
            
            # วาด text
            cv2.putText(
                frame_copy,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
            
            # วาด center point
            cx, cy = det['center']
            cv2.circle(frame_copy, (cx, cy), 3, color, -1)
        
        return frame_copy

if __name__ == "__main__":
    print("🧠 YOLO Detector Test")
    print("=" * 60)
    
    # สร้าง detector
    print("\n🔄 กำลังสร้าง YOLO detector...")
    detector = YOLODetector(
        confidence_threshold=0.5,
        nms_threshold=0.4
    )
    
    print(f"✅ ใช้ backend: {detector.backend}")
    print(f"📊 Vehicle classes: {list(detector.VEHICLE_CLASSES.keys())}")
    
    print("\n🎬 กำลังทดสอบกับ webcam...")
    print("💡 กด 'q' เพื่อออก | 's' เพื่อดู stats")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ ไม่สามารถเปิด webcam ได้")
    else:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            detections = detector.detect(frame)

            frame_with_boxes = detector.draw_detections(frame, detections)

            stats = detector.get_stats()
            info_text = f"Detections: {len(detections)} | FPS: {stats['fps']:.1f} | Backend: {stats['backend']}"
            cv2.putText(
                frame_with_boxes,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            cv2.imshow("YOLO Detector Test", frame_with_boxes)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                print(f"\n📊 Stats: {detector.get_stats()}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n📊 Final Stats:")
        for key, value in detector.get_stats().items():
            print(f"   {key}: {value}")
    
    print("\n✅ Test Complete!")