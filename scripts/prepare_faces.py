"""Extract frames from videos / images, detect, align and crop faces to 224x224 PNGs.

    python scripts/prepare_faces.py --input /data/raw/oulu --out /data/faces/oulu \\
        --detector-model models/face_detection_yunet_2023mar.onnx --frames-per-video 25

Output layout (consumed by scripts/build_manifest.py):
    <out>/<path of the video relative to --input, without extension>/<frame_index:05d>.png
Images found under --input are processed as single-frame "videos". Frames are sampled uniformly
(``--frames-per-video``); frames without a detected face are skipped. Existing outputs are kept, so
the command can be re-run to resume.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apr_tma.faces import YuNetDetector, align_crop  # noqa: E402

VIDEO_EXT = {".avi", ".mp4", ".mov", ".mkv", ".mpg", ".mpeg"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def frames_of(path: Path, n: int):
    if path.suffix.lower() in IMAGE_EXT:
        img = cv2.imread(str(path))
        if img is not None:
            yield 0, img
        return
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = set(np.linspace(0, max(total - 1, 0), num=min(n, max(total, 1)), dtype=int).tolist())
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in idx:
            yield i, frame
        i += 1
    cap.release()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--detector-model", required=True, help="YuNet ONNX file")
    ap.add_argument("--frames-per-video", type=int, default=25)
    ap.add_argument("--margin", type=float, default=1.3)
    ap.add_argument("--size", type=int, default=224)
    a = ap.parse_args()

    root, out = Path(a.input), Path(a.out)
    det = YuNetDetector(a.detector_model)
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXT | IMAGE_EXT)
    kept = missed = 0
    for p in tqdm(files, desc="media"):
        dest = out / p.relative_to(root).with_suffix("")
        for i, frame in frames_of(p, a.frames_per_video):
            target = dest / f"{i:05d}.png" if p.suffix.lower() in VIDEO_EXT else dest.with_suffix(".png")
            if target.exists():
                continue
            face = det(frame)
            if face is None:
                missed += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target), align_crop(frame, face, a.size, a.margin))
            kept += 1
    print(f"saved {kept} crops, {missed} frames without a detected face")


if __name__ == "__main__":
    main()
