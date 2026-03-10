# Import necessary libraries
import cv2
import os
from datetime import datetime
from PIL import Image
import torch
import json
import io
import sqlite3

import requests


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)


CONTRAST_ALPHA = 1.3  # >1.0 increases contrast
BRIGHTNESS_BETA = 0    # add/subtract brightness
USE_CLAHE = True       # adaptive contrast on luminance channel


def enhance_frame_bgr(frame_bgr):
    out = cv2.convertScaleAbs(frame_bgr, alpha=CONTRAST_ALPHA, beta=BRIGHTNESS_BETA)
    if not USE_CLAHE:
        return out

    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)



def send_img(path, mac="AS:AS:BS:AS:SD:AS", addr="Ahmedabad, Gota"):
    url = f'http://127.0.0.1:5000/add/{addr}/{mac}'

    with open(path, 'rb') as img:
        name_img = os.path.basename(path)
        files = {'file': (name_img, img, 'multipart/form-data', {'Expires': '0'})}
        with requests.Session() as s:
            r = s.post(url, files=files)
            print(r)


def get_yolov5():
    # Prefer local custom weights if present; otherwise fall back to a pretrained model
    # so the project can run on any machine without extra files.
    repo_dir = os.path.join(BASE_DIR, "yolov5")
    repo_or_dir = repo_dir if os.path.isdir(repo_dir) else "ultralytics/yolov5"

    # Custom garbage-detection weights live in Backend/model/best1000.pt
    weights_path = os.path.join(BASE_DIR, "model", "best1000.pt")
    if os.path.isfile(weights_path):
        model = torch.hub.load(
            repo_or_dir,
            "custom",
            path=weights_path,
            source="local" if os.path.isdir(repo_dir) else "github",
        )
    else:
        print(f"[INFO] Custom weights not found at: {weights_path}")
        print("[INFO] Falling back to pretrained 'yolov5s' weights.")
        model = torch.hub.load(
            repo_or_dir,
            "yolov5s",
            pretrained=True,
            source="local" if os.path.isdir(repo_dir) else "github",
        )

    # Thresholds:
    # - model.conf controls which boxes are returned/rendered by YOLO
    # - saving is additionally gated by SAVE_CONF_MIN below
    model.conf = 0.53
    model.iou = 0.45
    return model

model1 = get_yolov5()
# Directory in which frames and detected images will be stored
os.chdir(DATA_DIR)

# i variable is to give unique name to images
i = 1

# Save/send throttling (save at most once every N frames)
SAVE_EVERY_N_FRAMES = 60
DETECT_CONF_MIN = 0.53
SAVE_CONF_MIN = 0.62
frame_count = 0
x = 0
# Open the camera
video = cv2.VideoCapture(2)

db_path = os.path.join(BASE_DIR, "garbage.db")
conn = sqlite3.connect(db_path)

while True:
    # Read video by read() function and it
    # will extract and return the frame
    ret, frame = video.read()
    if not ret:
        break
    frame_count += 1

    # Make live feed look more natural + slightly higher contrast
    frame = enhance_frame_bgr(frame)

    # Put current DateTime on each frame
    font = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(
        frame,
        str(datetime.now()),
        (20, 40),
        font,
        2,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Default display is the raw frame; we may replace it with
    # a version that has YOLO bounding boxes drawn.
    display_frame = frame

    # Run detection on every frame (continuous)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model1(Image.fromarray(rgb), size=640)
    data = json.loads(results.pandas().xyxy[0].to_json(orient="records"))
    # `data` items include a `confidence` field from YOLO
    max_conf = max((d.get("confidence", 0.0) for d in data), default=0.0)

    # Render boxes onto the live view
    imgs = results.render()  # updates results.imgs with boxes and labels
    if imgs:
        # YOLO render outputs RGB; OpenCV expects BGR for display
        display_frame = cv2.cvtColor(imgs[0], cv2.COLOR_RGB2BGR)

    # Only save/send an image once every N frames, and only if confidence is high enough
    if (frame_count % SAVE_EVERY_N_FRAMES == 0) and (max_conf >= SAVE_CONF_MIN):
        print(f"Garbage detected (frame {frame_count}, conf={max_conf:.2f})")
        # Save as RGB for correct colors in output file
        annotated = imgs[0] if imgs else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detected_path = os.path.join(DATA_DIR, f"detected_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpeg")
        Image.fromarray(annotated).save(detected_path)
        send_img(detected_path, "LK:ASK:LAS:ASL")

    # Display the (possibly annotated) frame
    cv2.imshow("live video", display_frame)
    # wait for user to press any key (keep it responsive)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# close the camera
video.release()

# close open windows
cv2.destroyAllWindows()
