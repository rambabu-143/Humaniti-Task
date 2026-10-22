"""Video -> camera-tile signals (8x8 background, face, lighting), sampled every STEP seconds."""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

STEP = 30
W = 960
LOGO = cv2.imread(str(Path(__file__).parent / "assets" / "logo.png"), cv2.IMREAD_GRAYSCALE)
FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def sample(video, out):
    if list(out.glob("*.jpg")):
        return sorted(out.glob("*.jpg"))
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf", f"fps=1/{STEP},scale={W}:-1",
                    str(out / "%05d.jpg")], check=True)
    return sorted(out.glob("*.jpg"))


SCALES = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]  # tile size/position varies per class and Zoom layout


def analyse(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    best = 0.0
    for s in SCALES:
        t = cv2.resize(LOGO, None, fx=s, fy=s)
        best = max(best, cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED).max())
    # ponytail: whole-frame face search; screen-share frames only contain faces in the camera tile
    faces = FACE.detectMultiScale(g, 1.1, 6, minSize=(18, 18))
    bright = float(g[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]].mean()) if len(faces) else None
    return best > 0.5, len(faces) > 0, bright


def frame_features(video, cache, cues=None, teacher=None):
    """cues = features.parse() output. The tile shows the active speaker, so with cues
    only frames where the teacher was speaking (~6s before the frame) are scored."""
    res = []
    for i, p in enumerate(sample(Path(video), Path(cache))):
        t = i * STEP + STEP / 2
        if cues and not any(c[1] == teacher and t - 6 <= c[0] <= t for c in cues):
            continue
        res.append(analyse(cv2.imread(str(p))))
    n = len(res) or 1
    b = [r[2] for r in res if r[2] is not None]
    return {
        "frames": len(res),
        "logo_background_share": round(sum(r[0] for r in res) / n, 2),
        "face_visible_share": round(sum(r[1] for r in res) / n, 2),
        "face_brightness": round(float(np.mean(b)), 0) if b else None,  # 0-255; <70 is dim
    }


if __name__ == "__main__":
    for v in sys.argv[1:]:
        from features import features, parse
        code = Path(v).name.split()[0].upper()  # class code, e.g. B472, pairs video with its vtt
        vtt = next((x for x in Path(v).parent.glob("*.vtt") if x.name.upper().startswith(code)), None)
        c = Path("cache/frames") / Path(v).stem.replace(" ", "")
        f = features(vtt) if vtt else None
        print(v.split("/")[-2], code, frame_features(v, c, parse(vtt), f["teacher"]) if f else "no vtt")
