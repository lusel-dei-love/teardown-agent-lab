# ABOUTME: Live pixel observations: grabs the game window and downsamples to the small
# ABOUTME: RGB frame the pixel policy sees. This is the agent's ONLY view of the world.

from __future__ import annotations

import numpy as np

from teardown_lab.xdisplay import detect_display

# Small enough to train on and to keep the grab+resize inside the control-step budget;
# large enough that a 0.5 m block displacement is visible.
FRAME_W = 128
FRAME_H = 72


class FrameGrabber:
    """Captures downsampled RGB frames from an X display.

    Deliberately holds no game state: the policy's world knowledge must come through
    these pixels, never through the privileged bridge.
    """

    def __init__(self, display: str | None = None, width: int = FRAME_W, height: int = FRAME_H):
        self.display = display or detect_display()
        self.width = width
        self.height = height
        self._sct = None

    def _ensure(self):
        if self._sct is None:
            import mss

            self._sct = mss.mss(display=self.display)
        return self._sct

    def grab(self) -> np.ndarray:
        """One frame as uint8 (H, W, 3) RGB."""
        from PIL import Image

        sct = self._ensure()
        raw = sct.grab(sct.monitors[1])
        # mss yields BGRA; drop alpha and flip to RGB.
        arr = np.asarray(raw)[:, :, :3][:, :, ::-1]

        # Decimate by an integer stride before the real resize. Resizing straight from
        # 1920x1080 costs more than the screen grab itself; strided slicing is nearly
        # free and leaves PIL a much smaller image to filter.
        stride = max(1, min(arr.shape[0] // self.height, arr.shape[1] // self.width))
        if stride > 1:
            arr = arr[::stride, ::stride]

        img = Image.fromarray(arr).resize((self.width, self.height), Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)

    def close(self) -> None:
        if self._sct is not None:
            self._sct.close()
            self._sct = None


class FakeFrameGrabber:
    """Stand-in for tests.

    Deterministic by default so an env built on it steps reproducibly (gymnasium's
    env checker requires that). Set `vary=True` when a test needs frames to differ
    between calls.
    """

    def __init__(self, width: int = FRAME_W, height: int = FRAME_H, vary: bool = False):
        self.width = width
        self.height = height
        self.vary = vary
        self.calls = 0

    def grab(self) -> np.ndarray:
        self.calls += 1
        value = self.calls % 256 if self.vary else 128
        return np.full((self.height, self.width, 3), value, dtype=np.uint8)

    def close(self) -> None:
        pass
