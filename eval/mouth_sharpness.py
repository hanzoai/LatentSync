"""Mouth-region sharpness metric: Laplacian variance of the lower-face/mouth crop.
Compares a generated (dubbed) video against the original source video.
Reports mean mouth Laplacian variance for each and the % softening of generated vs source.
A lower generated value => softer/blurrier mouth. % softening = (1 - gen/src)*100.
Uses mediapipe FaceMesh to locate the mouth; falls back to a fixed lower-face box.
"""
import argparse, sys
import cv2
import numpy as np


def mouth_box(frame, face_mesh):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark
    # mouth landmark indices (outer lips region) in mediapipe facemesh
    idxs = [61, 291, 0, 17, 13, 14, 78, 308, 82, 312, 87, 317, 405, 181]
    xs = [lm[i].x * w for i in idxs]
    ys = [lm[i].y * h for i in idxs]
    x0, x1 = int(min(xs)), int(max(xs))
    y0, y1 = int(min(ys)), int(max(ys))
    # pad 30%
    pw = int((x1 - x0) * 0.3); ph = int((y1 - y0) * 0.3)
    x0 = max(0, x0 - pw); x1 = min(w, x1 + pw)
    y0 = max(0, y0 - ph); y1 = min(h, y1 + ph)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return x0, y0, x1, y1


def lap_var(crop):
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(g, cv2.CV_64F).var()


def measure(path, face_mesh, max_frames=400):
    cap = cv2.VideoCapture(path)
    vals = []
    n = 0
    while n < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        b = mouth_box(fr, face_mesh)
        if b is None:
            n += 1
            continue
        x0, y0, x1, y1 = b
        vals.append(lap_var(fr[y0:y1, x0:x1]))
        n += 1
    cap.release()
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True, help="generated/dubbed video")
    ap.add_argument("--src", required=True, help="original source video")
    args = ap.parse_args()
    import mediapipe as mp
    fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
    gv = measure(args.gen, fm)
    sv = measure(args.src, fm)
    if not gv or not sv:
        print("ERROR: no mouth detected", file=sys.stderr); sys.exit(1)
    g = float(np.mean(gv)); s = float(np.mean(sv))
    soft = (1 - g / s) * 100
    print(f"Source mouth Laplacian var (mean over {len(sv)} frames): {s:.2f}")
    print(f"Generated mouth Laplacian var (mean over {len(gv)} frames): {g:.2f}")
    print(f"Mouth softening (generated vs source): {soft:.1f}%  (positive = softer/blurrier)")


if __name__ == "__main__":
    main()
