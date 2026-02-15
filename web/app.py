import os
import cv2
import base64
import numpy as np
import tempfile

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

from DetectPerson import DetectPerson
from Encoder import Encoder
from tracker import Tracker


app = FastAPI()
templates = Jinja2Templates(directory="web/templates")

detector = DetectPerson()
encoder = Encoder()
tracker = Tracker()


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def process_video(file_path, tracker_instance):
    cap = cv2.VideoCapture(file_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    frame_index = 0
    appearances = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # process every 3rd frame
        if frame_index % 3 != 0:
            frame_index += 1
            continue

        detections = detector.detect(frame)

        for det in detections:
            crop = det["crop"]
            if crop is None or crop.size == 0:
                continue

            emb = encoder.encode(crop)
            pid = tracker_instance.match(emb)

            time_sec = frame_index / fps

            _, buffer = cv2.imencode(".png", crop)
            preview = base64.b64encode(buffer).decode()
            preview = f"data:image/png;base64,{preview}"

            appearances.setdefault(pid, []).append({
                "time": round(time_sec, 2),
                "preview": preview
            })

        frame_index += 1

    cap.release()
    return appearances


@app.post("/analyze")
async def analyze(
    video1: UploadFile = File(...),
    video2: UploadFile = File(None),
):
    tmp1 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp1.write(await video1.read())
    tmp1.close()

    # reset tracker for consistent results per request
    local_tracker = Tracker()

    appearances_v1 = process_video(tmp1.name, local_tracker)

    if video2 is None:
        os.remove(tmp1.name)

        recurring = []
        for pid, apps in appearances_v1.items():
            if len(apps) >= 2:
                recurring.append({
                    "id": pid,
                    "count": len(apps),
                    "appearances": apps
                })

        return {
            "video": "single",
            "recurring_persons": recurring
        }

    tmp2 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp2.write(await video2.read())
    tmp2.close()

    appearances_v2 = process_video(tmp2.name, local_tracker)

    os.remove(tmp1.name)
    os.remove(tmp2.name)

    common_ids = set(appearances_v1.keys()) & set(appearances_v2.keys())

    if common_ids:
        pid = list(common_ids)[0]

        return {
            "same_person": True,
            "person_id": pid,
            "appearances_video1": appearances_v1[pid],
            "appearances_video2": appearances_v2[pid],
        }

    return {"same_person": False}
