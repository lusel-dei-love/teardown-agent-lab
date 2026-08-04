# ABOUTME: Screen capture for eval artefacts: single-frame screenshots via mss and
# ABOUTME: per-episode MP4s via an ffmpeg x11grab subprocess (both lazily imported).

from __future__ import annotations

import subprocess
from pathlib import Path

DEFAULT_SIZE = (1920, 1080)


def screen_size(display: str, fallback: tuple[int, int] = DEFAULT_SIZE) -> tuple[int, int]:
    """Actual screen geometry. Hardcoding it silently breaks every recording when the
    resolution changes - ffmpeg refuses a capture area larger than the screen."""
    try:
        from Xlib import display as xdisplay

        conn = xdisplay.Display(display)
        try:
            root = conn.screen().root.get_geometry()
            return (int(root.width), int(root.height))
        finally:
            conn.close()
    except Exception:
        return fallback
DEFAULT_FPS = 30


def ffmpeg_cmd(display: str, size: tuple[int, int], fps: int, out: Path) -> list[str]:
    """Exact argv for an x11grab recording of `display` into `out`. Pure."""
    return [
        "ffmpeg",
        # Overwrite without prompting. Without this, a re-run against an existing path
        # leaves ffmpeg waiting on stdin and the old file in place - which reads as a
        # successful recording of the wrong episode.
        "-y",
        "-f",
        "x11grab",
        "-framerate",
        str(fps),
        "-video_size",
        f"{size[0]}x{size[1]}",
        "-i",
        display,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        str(out),
    ]


def screenshot(display: str, out_path: Path) -> Path:
    """Grab the full screen of `display` into `out_path` (PNG). Needs an X display."""
    import mss
    import mss.tools

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with mss.mss(display=display) as sct:
        frame = sct.grab(sct.monitors[1])
    mss.tools.to_png(frame.rgb, frame.size, output=str(out_path))
    return out_path


class EpisodeRecorder:
    """Starts/stops one ffmpeg screen recording per episode under `out_dir`."""

    def __init__(
        self,
        display: str,
        out_dir: Path,
        size: tuple[int, int] | None = None,
        fps: int = DEFAULT_FPS,
        popen=subprocess.Popen,
    ):
        self.display = display
        self.out_dir = Path(out_dir)
        self.size = size or screen_size(display)
        self.fps = fps
        self.popen = popen
        self._process = None
        self._path: Path | None = None

    def start(self, name: str) -> Path:
        if self._process is not None:
            raise RuntimeError("a recording is already in progress")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{name}.mp4"
        self._process = self.popen(ffmpeg_cmd(self.display, self.size, self.fps, path))
        self._path = path
        return path

    def stop(self) -> Path | None:
        """Terminate ffmpeg so it finalizes the file; returns the path, or None."""
        if self._process is None:
            return None
        self._process.terminate()
        self._process.wait()
        path, self._path, self._process = self._path, None, None
        return path
