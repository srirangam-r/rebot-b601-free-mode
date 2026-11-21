"""OpenCV UVC cameras (front + wrist), one background reader thread each, keeping only the latest frame
already centre-cropped to a square, resized to 256x256 and converted to RGB (the training format).

Sources: an integer index ("0"), a device path ("/dev/video2"), a stable path
("/dev/v4l/by-id/usb-...-video-index0"), or a video file (for testing).

  python -m b601free.cameras --list                     # list V4L2 devices (does not open them)
  python -m b601free.cameras --snap 0 /dev/v4l/by-id/X  # save one processed frame per source to /tmp/cam_*.png
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import threading
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)
OUT_SIZE = 256


def parse_source(s: str | int):
    if isinstance(s, int):
        return s
    s = str(s)
    return int(s) if s.isdigit() else s


def process(bgr: np.ndarray, size: int = OUT_SIZE) -> np.ndarray:
    """Centre crop to a square, resize to size x size, BGR -> RGB uint8."""
    h, w = bgr.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    sq = bgr[y0:y0 + s, x0:x0 + s]
    sq = cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA if s > size else cv2.INTER_LINEAR)
    return cv2.cvtColor(sq, cv2.COLOR_BGR2RGB)


class Camera:
    def __init__(self, name: str, source, width: int = 640, height: int = 480, fps: int = 30,
                 fourcc: str | None = "MJPG", size: int = OUT_SIZE):
        self.name, self.source = name, parse_source(source)
        self.width, self.height, self.fps, self.fourcc, self.size = width, height, fps, fourcc, size
        self.cap = None
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._raw: np.ndarray | None = None       # latest full-resolution BGR frame (for vision checks)
        self._t = 0.0
        self.n_frames = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _open(self):
        src = self.source
        if isinstance(src, str) and src.startswith("/dev/"):
            real = os.path.realpath(src)
            cap = cv2.VideoCapture(real, cv2.CAP_V4L2)
        elif isinstance(src, int):
            cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        else:
            cap = cv2.VideoCapture(src)  # video file
        if not cap.isOpened():
            raise RuntimeError(f"camera {self.name}: cannot open {src!r}")
        if not isinstance(src, str) or src.startswith("/dev/"):
            if self.fourcc:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        w, h, f = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FPS)
        logger.info("camera %s: %r opened at %dx%d @ %.0f fps", self.name, src, w, h, f)
        return cap

    def start(self, timeout: float = 5.0):
        self.cap = self._open()
        self._thread = threading.Thread(target=self._loop, name=f"cam-{self.name}", daemon=True)
        self._thread.start()
        t0 = time.time()
        while self._frame is None:
            if time.time() - t0 > timeout:
                self.stop()
                raise RuntimeError(f"camera {self.name}: no frame within {timeout}s from {self.source!r}")
            time.sleep(0.01)
        return self

    def _loop(self):
        fails = 0
        is_file = isinstance(self.source, str) and not self.source.startswith("/dev/")
        while not self._stop.is_set():
            ok, bgr = self.cap.read()
            if not ok or bgr is None:
                if is_file:  # loop test videos
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                fails += 1
                if fails == 50:
                    logger.warning("camera %s: no frames (unplugged?) - retrying quietly every 5 s", self.name)
                if fails >= 50:   # unplugged: back off, retry every ~5 s without flooding the terminal
                    time.sleep(0.1)
                if fails % 50 == 0:
                    try:
                        self.cap.release()
                        self.cap = self._open()
                        logger.warning("camera %s: reconnected", self.name)
                    except Exception:
                        pass
                time.sleep(0.01)
                continue
            fails = 0
            img = process(bgr, self.size)
            with self._lock:
                self._frame, self._raw, self._t = img, bgr, time.perf_counter()
                self.n_frames += 1
            if is_file:
                time.sleep(1.0 / max(self.fps, 1))

    def read(self) -> tuple[np.ndarray, float]:
        """Latest processed frame (256,256,3) uint8 RGB and its age in seconds."""
        with self._lock:
            return self._frame, time.perf_counter() - self._t

    def read_raw(self) -> tuple[np.ndarray | None, float]:
        """Latest full-resolution BGR frame and its age in seconds."""
        with self._lock:
            return self._raw, time.perf_counter() - self._t

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class CameraRig:
    """front + wrist. wrist=None and duplicate_front=True -> the front frame is used for both views."""

    def __init__(self, front, wrist=None, duplicate_front: bool = False, width=640, height=480, fps=30,
                 fourcc="MJPG", mock: bool = False, mock_frames: str | None = None):
        if wrist is None and not duplicate_front and not mock:
            raise ValueError("no wrist camera given: pass --wrist SRC or --duplicate_front")
        self.duplicate_front = duplicate_front or (wrist is None and not mock)
        if mock:
            from .mock import MockCamera
            self.front = MockCamera("front", "front", mock_frames)
            self.wrist = None if self.duplicate_front else MockCamera("wrist", "wrist", mock_frames)
        else:
            self.front = Camera("front", front, width, height, fps, fourcc)
            self.wrist = None if self.duplicate_front else Camera("wrist", wrist, width, height, fps, fourcc)
        self.max_age = 0.0

    def start(self):
        self.front.start()
        if self.wrist is not None:
            try:
                self.wrist.start()
            except Exception:
                self.front.stop()
                raise
        if self.duplicate_front:
            logger.warning("wrist view = duplicated front view (--duplicate_front): use the same flag at run time")
        return self

    def read(self):
        """-> front, wrist (256,256,3) uint8 RGB, (front_age_s, wrist_age_s)."""
        f, fa = self.front.read()
        if self.wrist is None:
            w, wa = f, fa
        else:
            w, wa = self.wrist.read()
        self.max_age = max(fa, wa)
        return f, w, (fa, wa)

    def info(self) -> dict:
        return {"front": str(self.front.__dict__.get("source", "mock")),
                "wrist": "duplicate_front" if self.duplicate_front else str(self.wrist.__dict__.get("source", "mock"))}

    def stop(self):
        for c in (self.front, self.wrist):
            if c is not None:
                c.stop()


def add_camera_args(p: argparse.ArgumentParser):
    p.add_argument("--front", default="0", help="front camera: index, /dev/videoN or /dev/v4l/by-id/... path")
    p.add_argument("--wrist", default=None, help="wrist camera (same forms as --front)")
    p.add_argument("--duplicate_front", action="store_true", help="no wrist camera: use the front view twice")
    p.add_argument("--cam_width", type=int, default=640)
    p.add_argument("--cam_height", type=int, default=480)
    p.add_argument("--cam_fps", type=int, default=30)
    p.add_argument("--cam_fourcc", default="MJPG", help="'' to keep the driver default")
    p.add_argument("--mock_frames", default=None, help="mock: replay front/wrist from an ep_*.npz (file or dir)")


def make_rig(args) -> CameraRig:
    return CameraRig(args.front, args.wrist, args.duplicate_front, args.cam_width, args.cam_height, args.cam_fps,
                     args.cam_fourcc or None, mock=args.mock, mock_frames=args.mock_frames)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true")
    p.add_argument("--snap", nargs="*", default=None, help="sources to open and save one frame from")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    if a.list or a.snap is None:
        for path in sorted(glob.glob("/dev/v4l/by-id/*")):
            print(f"{path} -> {os.path.realpath(path)}")
        print("devices:", " ".join(sorted(glob.glob("/dev/video*"))))
        print("note: each UVC camera usually exposes 2 nodes; the capture one is normally the *-index0 link.")
    for i, s in enumerate(a.snap or []):
        cam = Camera(f"c{i}", s).start()
        time.sleep(0.5)
        img, age = cam.read()
        out = f"/tmp/cam_{i}.png"
        cv2.imwrite(out, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"{s}: saved {out} (age {age * 1e3:.0f} ms, {cam.n_frames} frames)")
        cam.stop()


if __name__ == "__main__":
    main()
