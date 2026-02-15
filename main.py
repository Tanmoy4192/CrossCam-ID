import threading
import cv2

from Camera import Camera
from DetectPerson import DetectPerson
from Encoder import Encoder
from tracker import Tracker


# Shared tracker across all cameras
tracker = Tracker()

# Stop signal shared between threads
stop_event = threading.Event()


def run_camera(cam_id, cam_index):
    camera = Camera(cam_index)
    detector = DetectPerson()
    encoder = Encoder()

    while not stop_event.is_set():
        frame = camera.get_frame()
        if frame is None:
            break

        detections = detector.detect(frame)

        for det in detections:
            crop = det["crop"]
            if crop is None or crop.size == 0:
                continue

            embedding = encoder.encode(crop)
            person_id = tracker.match(embedding)

            x1, y1, x2, y2 = det["bbox"]

            # draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # draw ID label
            cv2.putText(
                frame,
                person_id,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.imshow(f"Camera {cam_id}", frame)

        # exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()
            break

    camera.release()


def main():
    threads = [
        threading.Thread(target=run_camera, args=(0, 0)),
        threading.Thread(target=run_camera, args=(1, "http://10.131.168.245:8000/video")),
    ] #IP camera/ use 1 if using any secondary device with usb

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
