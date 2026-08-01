# ABOUTME: Tests for the Actuator's action -> input mapping against RecordingBackend.
# ABOUTME: Headless by construction: no pyautogui, no X display, no game.

import subprocess
import sys

from teardown_lab.actuator import Action, Actuator, InputBackend, RecordingBackend


def make() -> tuple[Actuator, RecordingBackend]:
    backend = RecordingBackend()
    return Actuator(backend), backend


def test_recording_backend_satisfies_protocol():
    assert isinstance(RecordingBackend(), InputBackend)


def test_importing_actuator_does_not_import_pyautogui():
    code = "import teardown_lab.actuator, sys; print('pyautogui' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"


def test_look_scales_to_pixels():
    actuator, backend = make()
    actuator.apply(Action(look_dx=0.5, look_dy=-0.25))
    assert ("move_mouse", 100, -50) in backend.calls


def test_look_scale_is_configurable():
    backend = RecordingBackend()
    Actuator(backend, look_scale=10).apply(Action(look_dx=1.0, look_dy=1.0))
    assert ("move_mouse", 10, 10) in backend.calls


def test_move_forward_holds_w():
    actuator, backend = make()
    actuator.apply(Action(move_y=0.9))
    assert ("key", "w", True) in backend.calls


def test_held_key_is_not_pressed_twice():
    actuator, backend = make()
    actuator.apply(Action(move_y=0.9))
    backend.calls.clear()
    actuator.apply(Action(move_y=0.5))
    assert backend.key_calls == []


def test_key_released_when_input_drops_below_threshold():
    actuator, backend = make()
    actuator.apply(Action(move_y=0.9))
    backend.calls.clear()
    actuator.apply(Action(move_y=0.3))
    assert backend.key_calls == [("key", "w", False)]


def test_direction_mapping():
    cases = {
        "w": Action(move_y=1.0),
        "s": Action(move_y=-1.0),
        "d": Action(move_x=1.0),
        "a": Action(move_x=-1.0),
    }
    for expected, action in cases.items():
        actuator, backend = make()
        actuator.apply(action)
        assert backend.key_calls == [("key", expected, True)]


def test_deadzone_presses_nothing():
    actuator, backend = make()
    actuator.apply(Action(move_x=0.3, move_y=-0.3))
    assert backend.key_calls == []


def test_opposite_direction_swaps_keys():
    actuator, backend = make()
    actuator.apply(Action(move_y=1.0))
    backend.calls.clear()
    actuator.apply(Action(move_y=-1.0))
    assert backend.key_calls == [("key", "w", False), ("key", "s", True)]


def test_buttons_follow_grab_and_swing():
    actuator, backend = make()
    actuator.apply(Action(grab=True, swing=True))
    assert backend.button_calls == [
        ("button", "left", True),
        ("button", "right", True),
    ]
    backend.calls.clear()
    actuator.apply(Action(grab=True, swing=False))
    assert backend.button_calls == [("button", "left", False)]


def test_release_all_releases_exactly_the_held_set():
    actuator, backend = make()
    actuator.apply(Action(move_x=-1.0, move_y=1.0, grab=True))
    backend.calls.clear()
    actuator.release_all()
    assert set(backend.key_calls) == {("key", "a", False), ("key", "w", False)}
    assert backend.button_calls == [("button", "right", False)]
    backend.calls.clear()
    actuator.release_all()
    assert backend.calls == []
