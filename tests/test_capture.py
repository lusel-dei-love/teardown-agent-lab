# ABOUTME: Tests for the capture module: the pure ffmpeg argv builder and the recorder
# ABOUTME: lifecycle driven through an injected fake popen (no ffmpeg, no X display).

import subprocess
import sys

import pytest

from teardown_lab.capture import EpisodeRecorder, ffmpeg_cmd


class FakeProcess:
    def __init__(self):
        self.terminated = False
        self.waited = False

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        self.waited = True
        return 0


class FakePopen:
    def __init__(self):
        self.calls: list[list[str]] = []
        self.processes: list[FakeProcess] = []

    def __call__(self, argv):
        self.calls.append(list(argv))
        process = FakeProcess()
        self.processes.append(process)
        return process


def test_ffmpeg_cmd_exact_argv(tmp_path):
    out = tmp_path / "ep_001.mp4"
    assert ffmpeg_cmd(":1", (1920, 1080), 30, out) == [
        "ffmpeg",
        "-y",
        "-f",
        "x11grab",
        "-framerate",
        "30",
        "-video_size",
        "1920x1080",
        "-i",
        ":1",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        str(out),
    ]


def test_ffmpeg_cmd_honours_parameters(tmp_path):
    argv = ffmpeg_cmd(":7", (800, 600), 15, tmp_path / "a.mp4")
    assert argv[argv.index("-framerate") + 1] == "15"
    assert argv[argv.index("-video_size") + 1] == "800x600"
    assert argv[argv.index("-i") + 1] == ":7"


def test_importing_capture_does_not_import_mss():
    code = "import teardown_lab.capture, sys; print('mss' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"


def test_start_spawns_ffmpeg_into_out_dir(tmp_path):
    popen = FakePopen()
    out_dir = tmp_path / "videos"
    recorder = EpisodeRecorder(":1", out_dir, popen=popen)
    path = recorder.start("ep_003")
    assert path == out_dir / "ep_003.mp4"
    assert out_dir.is_dir()
    assert popen.calls == [ffmpeg_cmd(":1", recorder.size, recorder.fps, path)]


def test_stop_terminates_and_waits(tmp_path):
    popen = FakePopen()
    recorder = EpisodeRecorder(":1", tmp_path, popen=popen)
    path = recorder.start("ep_001")
    assert recorder.stop() == path
    process = popen.processes[0]
    assert process.terminated is True
    assert process.waited is True


def test_stop_without_start_is_noop(tmp_path):
    recorder = EpisodeRecorder(":1", tmp_path, popen=FakePopen())
    assert recorder.stop() is None


def test_start_while_recording_raises(tmp_path):
    recorder = EpisodeRecorder(":1", tmp_path, popen=FakePopen())
    recorder.start("ep_001")
    with pytest.raises(RuntimeError):
        recorder.start("ep_002")


def test_start_after_stop_records_again(tmp_path):
    popen = FakePopen()
    recorder = EpisodeRecorder(":1", tmp_path, popen=popen)
    recorder.start("ep_001")
    recorder.stop()
    recorder.start("ep_002")
    assert [call[-1] for call in popen.calls] == [
        str(tmp_path / "ep_001.mp4"),
        str(tmp_path / "ep_002.mp4"),
    ]
