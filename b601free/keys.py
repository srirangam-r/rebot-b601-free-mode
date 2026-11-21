"""Non-blocking single-key input from the terminal (cbreak mode, Ctrl-C still raises KeyboardInterrupt).
Also works when stdin is a pipe, e.g. `(sleep 2; printf r; sleep 3; printf s; printf q) | python -m ...`."""

from __future__ import annotations

import os
import select
import sys

_NAMES = {" ": "space", "\n": "enter", "\r": "enter", "\x1b": "esc"}


class KeyReader:
    def __init__(self):
        self.fd = sys.stdin.fileno() if sys.stdin is not None else None
        self.tty = self.fd is not None and os.isatty(self.fd)
        self.eof = self.fd is None
        self._old = None
        if self.tty:
            import termios
            import tty
            self._old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)

    def get(self) -> list[str]:
        """All keys pressed since the last call ('space', 'enter', 'esc' or the lower-case character)."""
        if self.eof:
            return []
        keys = []
        while True:
            r, _, _ = select.select([self.fd], [], [], 0)
            if not r:
                break
            data = os.read(self.fd, 64)
            if not data:
                self.eof = True
                break
            for ch in data.decode(errors="ignore"):
                keys.append(_NAMES.get(ch, ch.lower()))
        return keys

    def close(self):
        if self._old is not None:
            import termios
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._old)
            self._old = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
