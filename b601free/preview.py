"""On-screen preview of both camera views in a separate process (Tk + Pillow; the venv's OpenCV is
headless). The control loop only does a non-blocking queue put, so a slow window never stalls it."""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue

import numpy as np

logger = logging.getLogger(__name__)


def _proc(q, title: str, scale: float):
    import tkinter as tk
    from PIL import Image, ImageTk
    root = tk.Tk()
    root.title(title)
    img_label = tk.Label(root)
    img_label.pack()
    txt = tk.Label(root, font=("monospace", 11), justify="left", anchor="w")
    txt.pack(fill="x")
    state = {"open": True}

    def poll():
        item = None
        try:
            while True:
                item = q.get_nowait()
                if item is None:
                    root.destroy()
                    return
        except queue.Empty:
            pass
        if item is not None:
            front, wrist, text = item
            both = np.concatenate([front, wrist], 1)
            im = Image.fromarray(both)
            if scale != 1.0:
                im = im.resize((int(im.width * scale), int(im.height * scale)))
            photo = ImageTk.PhotoImage(im)
            img_label.configure(image=photo)
            img_label.image = photo
            txt.configure(text=text)
        if state["open"]:
            root.after(25, poll)

    root.after(25, poll)
    root.mainloop()


class Preview:
    def __init__(self, enabled: bool = True, title: str = "front | wrist", scale: float = 1.5, every: int = 3):
        self.enabled = enabled and bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if enabled and not self.enabled:
            logger.info("no display: preview disabled")
        self.title, self.scale, self.every = title, scale, every
        self.p = None
        self.q = None
        self._n = 0

    def start(self):
        if not self.enabled:
            return self
        try:
            ctx = mp.get_context("spawn")
            self.q = ctx.Queue(maxsize=2)
            self.p = ctx.Process(target=_proc, args=(self.q, self.title, self.scale), daemon=True)
            self.p.start()
        except Exception as e:
            logger.warning("preview failed to start: %s", e)
            self.enabled = False
        return self

    def show(self, front, wrist, text: str = ""):
        """Call every control tick; only every `every`-th frame is sent (30 Hz -> 10 Hz)."""
        self._n += 1
        if not self.enabled or self._n % self.every or self.p is None or not self.p.is_alive():
            return
        try:
            self.q.put_nowait((front, wrist, text))
        except queue.Full:
            pass

    def close(self):
        if self.p is None:
            return
        try:
            self.q.put(None, timeout=0.5)
        except Exception:
            pass
        self.p.join(timeout=2.0)
        if self.p.is_alive():
            self.p.terminate()
        self.p = None
