from ultralytics import YOLO
import cv2
#import numpy as np
import torch

class DetectPerson:
    def __init__(self, model_name = "yolov8n.pt", conf = 0.78):
        #self.device = torch.device("cuda : 0" if torch.cuda.is_available() else "cpu")
        print(f"loading model : {model_name}...")
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_name).to(self.device) #load  model 
        self.conf = conf
        self.person_class_id = 0 #COCO class 0 = person

    def detect(self,frame):
        results = self.model.predict(
            source = frame,
            iou = 0.45,
            conf = self.conf,
            classes = [self.person_class_id],
            verbose = False,
        )
        detections = []
        min_width, min_height = 40, 80
        for r in results[0].boxes:
            if int(r.cls) != self.person_class_id:
                continue
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            width = x2 - x1
            height = y2 - y1

            if width < min_width or height < min_height:
                continue  # skip tiny false positives

            conf_val = float(r.conf[0])
            crop = frame[y1:y2, x1:x2].copy() if (height > 0 and width > 0) else None

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "conf": conf_val,
                "crop": crop
            })
        return detections
    
    def draw_detections(self,frame,detections):
        for det in detections:
            x1,y1,x2,y2 = det["bbox"]
            conf = det["conf"]
            label = f"person{conf:.2f}"

            cv2.rectangle(frame,(x1,y1),(x2,y2), (0,255,0), 2)
            cv2.putText(frame, label,(x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        return frame


