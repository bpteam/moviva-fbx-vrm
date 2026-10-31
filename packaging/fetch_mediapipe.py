"""Downloads the pinned MediaPipe Face Landmarker model into ``vendor/mediapipe`` (build time only).

The model (Apache-2.0, Google) finds 478 face points on a rendered view of the head; see
``character_vrm/blender/face_detect.py``. The ``mediapipe`` wheel itself is installed with ``--no-deps``: only its
drawing helpers need OpenCV and matplotlib, and they are never used.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
TARGET = Path(__file__).resolve().parents[1] / "vendor" / "mediapipe" / "face_landmarker.task"


def main() -> int:
    data = urllib.request.urlopen(URL, timeout=120).read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SHA256:
        print(f"ERROR {URL}: sha256 {digest}, expected {SHA256}", file=sys.stderr)
        return 1
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_bytes(data)
    print(f"MediaPipe Face Landmarker -> {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
