import cv2

class Camera:
    def __init__(self,index = 0):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera.")
    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frames from camera.")
        return frame
    
    def release(self):
        self.cap.release()