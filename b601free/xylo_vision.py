"""Overhead-camera xylophone bar detection. Used ONLY to (a) place bars 2-7 between hand-taught anchor bars and
(b) check the xylophone hasn't moved since teaching. It never decides where the arm goes on its own: every pose it
produces is a blend of two hand-taught poses (inside the taught box). Detection thresholds tuned on a toy 8-bar xylophone under an overhead webcam."""

from __future__ import annotations

import cv2
import numpy as np


def detect_bars(bgr: np.ndarray, roi=(260, 50, 640, 420), n: int = 8):
    """-> list of n (cx, cy, width_px), bar 1 (longest) first, or None if not exactly n bars are found.
    roi = (x0, y0, x1, y1): search window in the 640x480 overhead image."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    x0, y0, x1, y1 = roi
    med = np.median(lab[y0:y1, x0:x1].reshape(-1, 3), axis=0)            # table colour
    dist_ab = np.linalg.norm(lab[:, :, 1:3] - med[1:3], axis=2)
    gx = cv2.Sobel(lab, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lab, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt((gx ** 2 + gy ** 2).sum(axis=2))
    mask = (((dist_ab > 26) | (hsv[:, :, 1] > 165)) & (hsv[:, :, 2] > 115) & (grad < 45)).astype(np.uint8) * 255
    keep = np.zeros_like(mask)
    keep[y0:y1, x0:x1] = 1
    mask *= keep
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    num, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
    bars = []
    for i in range(1, num):
        area, w, h = stats[i, cv2.CC_STAT_AREA], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if 250 < area < 2500 and w > 35 and h < 60:
            ys, xs = np.where(labels == i)
            (_, _), (rw, rh), _ = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
            bars.append((float(cents[i][0]), float(cents[i][1]), float(max(rw, rh))))   # (cx, cy, bar length px)
    if len(bars) != n:
        return None
    c = np.array([(x, y) for x, y, _ in bars])
    axis = np.linalg.svd(c - c.mean(0))[2][0]            # the xylophone's long axis in the image
    bars.sort(key=lambda b_: float(np.dot((b_[0], b_[1]), axis)))
    if np.mean([b_[2] for b_ in bars[:2]]) < np.mean([b_[2] for b_ in bars[-2:]]):
        bars.reverse()                                     # bar 1 = the longest bars' end
    return bars


def fractions(bars):
    """Position of every bar centre along the bar1 -> barN line, 0..1."""
    c = np.array([(b[0], b[1]) for b in bars])
    d = c[-1] - c[0]
    return np.clip(((c - c[0]) @ d) / (d @ d), 0.0, 1.0)


def max_shift(bars, ref):
    """How far the xylophone moved vs ref [px], robust to partly-detected bars (which slide along their length):
    max of (median translation of all bars, largest shift ALONG the row of bars, rotation of the row x half its
    length). Order-independent (matched to ref)."""
    bars = match(bars, ref)
    A = np.array([(b_[0], b_[1]) for b_ in bars]); R = np.array([(b_[0], b_[1]) for b_ in ref])
    d = A - R
    axis = (R[-1] - R[0]) / np.linalg.norm(R[-1] - R[0])
    trans = float(np.linalg.norm(np.median(d, 0)))
    along = float(np.abs(d @ axis).max())
    ang = lambda P: np.arctan2(*(np.linalg.svd(P - P.mean(0))[2][0][::-1]))   # row direction
    dth = abs((ang(A) - ang(R) + np.pi / 2) % np.pi - np.pi / 2)                # line angle, mod 180 deg
    rot = float(dth * np.linalg.norm(R[-1] - R[0]) / 2)
    return max(trans, along, rot)


def draw(bgr, bars, label=""):
    vis = bgr.copy()
    for i, (u, v, _) in enumerate(bars, 1):
        cv2.circle(vis, (int(u), int(v)), 5, (0, 255, 0), -1)
        cv2.putText(vis, str(i), (int(u) + 8, int(v) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    if label:
        cv2.putText(vis, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return vis


def match(a, ref):
    """Reorder detection a to the stored ref (identity or reversed, whichever is closer)."""
    A = np.array([(x, y) for x, y, *_ in a]); R = np.array([(x, y) for x, y, *_ in ref])
    if np.linalg.norm(A[::-1] - R, axis=1).mean() < np.linalg.norm(A - R, axis=1).mean():
        return list(a)[::-1]
    return list(a)


class BarWatcher:
    """Background detection (~6 Hz) on a threaded camera; consensus() = median of recent agreeing detections."""

    def __init__(self, cam, period: float = 0.15):
        import threading
        self.cam, self.period, self.dets = cam, period, []
        self._lock, self._stop = threading.Lock(), threading.Event()
        self._last_frame = None
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def _loop(self):
        import time
        while not self._stop.is_set():
            try:
                f, age = self.cam.read_raw()
                if f is not None and age < 0.5:
                    b = detect_bars(f)
                    with self._lock:
                        self._last_frame = f
                        if b is not None:
                            self.dets = [d for d in self.dets if time.perf_counter() - d[0] < 3.0] + [(time.perf_counter(), b)]
            except Exception:
                pass
            time.sleep(self.period)

    def consensus(self, max_age: float = 1.5, min_n: int = 3):
        """-> (bars, n_used, frame) from detections in the last max_age s, or (None, n, frame)."""
        import time
        with self._lock:
            recent = [b for t, b in self.dets if time.perf_counter() - t < max_age]
            frame = self._last_frame
        if len(recent) < min_n:
            return None, len(recent), frame
        base = recent[-1]
        al = [match(b, base) for b in recent]
        C = np.median(np.array([[(x, y) for x, y, _ in b] for b in al]), 0)
        L = np.median(np.array([[ln for *_, ln in b] for b in al]), 0)
        bars = [(float(x), float(y), float(ln)) for (x, y), ln in zip(C, L)]
        if np.mean(L[:2]) < np.mean(L[-2:]):
            bars.reverse()
        return bars, len(recent), frame

    def stop(self):
        self._stop.set()
