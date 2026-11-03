"""Downloads the pinned MediaPipe models into ``vendor/mediapipe`` (build time only): the Face Landmarker and the
multi-class selfie segmenter.

The model (Apache-2.0, Google) finds 478 face points on a rendered view of the head; see
``character_vrm/blender/face_detect.py``. The ``mediapipe`` wheel itself is installed with ``--no-deps``: only its
drawing helpers need OpenCV and matplotlib, and they are never used.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

MODELS = {  # file -> (url, sha256)
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"),
    # skin / clothes / hair per pixel, for materials whose names say nothing (blender/surface_vote.py)
    "selfie_multiclass.tflite": (
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/"
        "selfie_multiclass_256x256.tflite",
        "c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0"),
}
TARGET = Path(__file__).resolve().parents[1] / "vendor" / "mediapipe"


def main() -> int:
    TARGET.mkdir(parents=True, exist_ok=True)
    for name, (url, sha) in MODELS.items():
        data = urllib.request.urlopen(url, timeout=120).read()
        digest = hashlib.sha256(data).hexdigest()
        if digest != sha:
            print(f"ERROR {url}: sha256 {digest}, expected {sha}", file=sys.stderr)
            return 1
        (TARGET / name).write_bytes(data)
        print(f"MediaPipe {name} -> {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
