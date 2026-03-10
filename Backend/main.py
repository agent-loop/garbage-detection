# Basic YOLO-based garbage detection script.
# - Opens a webcam
# - Runs YOLOv5 on every frame
# - Shows live bounding boxes
# - Saves an annotated image every N frames when garbage is detected

import cv2
import os
from datetime import datetime
from PIL import Image
import torch
import json


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)


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

    # Only keep reasonably confident detections
    model.conf = 0.5
    model.iou = 0.45
    return model

model1 = get_yolov5()
# Directory in which frames and detected images will be stored
os.chdir(DATA_DIR)

# Save throttling (save at most once every N frames)
SAVE_EVERY_N_FRAMES = 60
frame_count = 0

def main():
    global frame_count

    # Open the default camera (0); change if needed
    video = cv2.VideoCapture(0)

    if not video.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame_count += 1

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

        # Run detection on every frame (continuous)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model1(Image.fromarray(rgb), size=640)
        detections = json.loads(results.pandas().xyxy[0].to_json(orient="records"))
        max_conf = max((d.get("confidence", 0.0) for d in detections), default=0.0)

        # Render boxes onto the live view
        imgs = results.render()  # list of RGB arrays
        display_frame = frame
        if imgs:
            display_frame = cv2.cvtColor(imgs[0], cv2.COLOR_RGB2BGR)

        # Save an annotated image every N frames if we saw at least one detection
        if detections and (frame_count % SAVE_EVERY_N_FRAMES == 0):
            print(f"Garbage detected (frame {frame_count}, max_conf={max_conf:.2f})")
            annotated_rgb = imgs[0] if imgs else rgb
            detected_path = os.path.join(
                DATA_DIR, f"detected_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpeg"
            )
            Image.fromarray(annotated_rgb).save(detected_path)

        cv2.imshow("live video", display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
