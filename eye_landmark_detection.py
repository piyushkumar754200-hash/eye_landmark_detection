"""
Eye Landmark Detection — Terminal / Standalone Version
------------------------------------------------------
Same detection logic as Eye_landmark_detection_jupyter.py, but displays in a
native OpenCV window instead of a Jupyter widget — so it runs as a normal
script:  python eye_landmark_detection_cv2.py

Press  q  or  ESC  in the video window to quit.

Requires camera permission for the terminal/IDE you launch this from
(macOS: System Settings -> Privacy & Security -> Camera).

Model (~4 MB) is auto-downloaded on first run.
"""

import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os
import sys
import time

# Eye landmark indices (MediaPipe 478-point face mesh)
LEFT_EYE   = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
RIGHT_EYE  = [33,  7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EAR_PTS  = [362, 385, 387, 263, 373, 380]
RIGHT_EAR_PTS = [33,  160, 158, 133, 153, 144]

EAR_THRESHOLD    = 0.25
BLINK_MIN_FRAMES = 2


def download_model(path: str = "face_landmarker.task") -> str:
    if not os.path.exists(path):
        print("Downloading face_landmarker model (~4 MB) ...")
        url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )
        urllib.request.urlretrieve(url, path)
        print("Download complete.")
    return path


def eye_aspect_ratio(landmarks, six_pts, w, h):
    p = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in six_pts]
    v1 = np.linalg.norm(np.array(p[1]) - np.array(p[5]))
    v2 = np.linalg.norm(np.array(p[2]) - np.array(p[4]))
    hz = np.linalg.norm(np.array(p[0]) - np.array(p[3]))
    return (v1 + v2) / (2.0 * hz + 1e-6)


def draw_eye(frame, landmarks, indices, color, w, h):
    pts = np.array(
        [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices],
        dtype=np.int32,
    )
    for pt in pts:
        cv2.circle(frame, tuple(pt), 2, color, -1, cv2.LINE_AA)
    hull = cv2.convexHull(pts)
    cv2.polylines(frame, [hull], True, color, 1, cv2.LINE_AA)


def draw_iris(frame, landmarks, indices, color, w, h):
    pts = np.array(
        [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices],
        dtype=np.int32,
    )
    center = pts.mean(axis=0).astype(int)
    radius = int(np.linalg.norm(pts[0] - pts[2]) / 2)
    cv2.circle(frame, tuple(center), radius, color, 1, cv2.LINE_AA)
    cv2.circle(frame, tuple(center), 2,      color, -1, cv2.LINE_AA)


def draw_hud(frame, ear, blinks, blink_l, blink_r):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (270, 140), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    def txt(s, y, color=(255, 255, 255)):
        cv2.putText(frame, s, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)

    txt(f"EAR:       {ear:.3f}", 28)
    txt(f"Blinks:    {blinks}",  58)
    txt(f"Blink L:   {blink_l:.2f}", 88,
        (0, 255, 0) if blink_l > 0.5 else (160, 160, 160))
    txt(f"Blink R:   {blink_r:.2f}", 118,
        (0, 180, 255) if blink_r > 0.5 else (160, 160, 160))
    txt("press q / ESC to quit", 138, (200, 200, 200))


def main():
    model_path = download_model()

    BaseOptions           = mp.tasks.BaseOptions
    FaceLandmarker        = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode           = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
    )

    # On macOS, force the AVFoundation backend. The default backend probes
    # FFMPEG/GStreamer first and can hang for a long time at VideoCapture(),
    # especially while the OS camera-permission prompt is pending.
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY

    # Camera index: pass one explicitly to avoid macOS Continuity Camera
    # (your iPhone) which usually claims index 0. The Mac's built-in webcam
    # is typically index 1.
    #   python eye_landmark_detection_cv2.py 1
    if len(sys.argv) > 1:
        indices = [int(sys.argv[1])]
    else:
        indices = [0, 1, 2]

    cap = None
    for cam_index in indices:
        print(f"Trying camera index {cam_index} ...")
        test = cv2.VideoCapture(cam_index, backend)
        if test.isOpened():
            cap = test
            print(f"Camera opened on index {cam_index}")
            break
        test.release()

    if cap is None:
        print("ERROR: No camera found / not authorized on indices 0, 1, 2.")
        print("  - Make sure your webcam is plugged in.")
        print("  - macOS: grant Camera permission to your terminal/IDE in")
        print("    System Settings -> Privacy & Security -> Camera, then retry.")
        return

    blink_counter = 0
    total_blinks  = 0
    t0 = time.time()

    try:
        with FaceLandmarker.create_from_options(options) as detector:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Frame read failed — check camera connection.")
                    break

                h, w  = frame.shape[:2]
                ts_ms = int((time.time() - t0) * 1000)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = detector.detect_for_video(mp_img, ts_ms)

                if result.face_landmarks:
                    lms = result.face_landmarks[0]

                    draw_eye(frame, lms, LEFT_EYE,  (0, 255,   0), w, h)
                    draw_eye(frame, lms, RIGHT_EYE, (0, 180, 255), w, h)
                    draw_iris(frame, lms, LEFT_IRIS,  (0, 255,   0), w, h)
                    draw_iris(frame, lms, RIGHT_IRIS, (0, 180, 255), w, h)

                    l_ear = eye_aspect_ratio(lms, LEFT_EAR_PTS,  w, h)
                    r_ear = eye_aspect_ratio(lms, RIGHT_EAR_PTS, w, h)
                    avg   = (l_ear + r_ear) / 2.0

                    if avg < EAR_THRESHOLD:
                        blink_counter += 1
                    else:
                        if blink_counter >= BLINK_MIN_FRAMES:
                            total_blinks += 1
                        blink_counter = 0

                    blink_l = blink_r = 0.0
                    if result.face_blendshapes:
                        bs_map = {b.category_name: b.score
                                  for b in result.face_blendshapes[0]}
                        blink_l = bs_map.get("eyeBlinkLeft",  0.0)
                        blink_r = bs_map.get("eyeBlinkRight", 0.0)

                    draw_hud(frame, avg, total_blinks, blink_l, blink_r)

                # Flip for mirror view and show in a native window.
                display_frame = cv2.flip(frame, 1)
                cv2.imshow("Eye Landmark Detection (q / ESC to quit)", display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # q or ESC
                    break

    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released.")


if __name__ == "__main__":
    main()