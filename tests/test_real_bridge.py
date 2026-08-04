# ABOUTME: Tests for RealBridge staleness handling against a synthetic savegame file.
# ABOUTME: No game required - the transport is just a file the mod rewrites.

import os

from teardown_lab.real_bridge import RealBridge


def write_payload(path, seq, episode=0, phase=0):
    """Write a payload, forcing a distinct mtime.

    The bridge skips re-reading a file whose mtime is unchanged - a real optimisation,
    since the game rewrites this file every frame. Test writes land inside one filesystem
    timestamp tick, so the mtime must be advanced explicitly.
    """
    blocks = "0.000,0.000,0.000"
    payload = (
        f"{seq}|1.000|{episode}|{episode}|1.000,2.000,3.000|0.000,0.000|{phase}|"
        f"{blocks}|{blocks}|{blocks}"
    )
    path.write_text(
        "<registry><savegame><mod><local-teardownlab>"
        f'<state value="{payload}"/>'
        "</local-teardownlab></mod></savegame></registry>",
        encoding="utf-8",
    )
    stamp = path.stat().st_mtime_ns + seq * 1_000_000
    os.utime(path, ns=(stamp, stamp))


def make_bridge(tmp_path):
    savegame = tmp_path / "savegame.xml"
    mod_dir = tmp_path / "mod"
    mod_dir.mkdir()
    return (
        RealBridge(savegame=savegame, mod_dir=mod_dir, sleeper=lambda _: None),
        savegame,
    )


def test_repeated_frame_is_not_returned_twice(tmp_path):
    bridge, savegame = make_bridge(tmp_path)
    write_payload(savegame, seq=5)
    assert bridge.read_state(timeout=0.01) is not None
    # Same seq, touched file: still the same frame.
    write_payload(savegame, seq=5)
    assert bridge.read_state(timeout=0.01) is None


def test_counter_going_backwards_is_a_restart_not_a_stale_frame(tmp_path):
    # savegame.xml survives across sessions. After a level reload the mod's counter
    # starts again at 1; treating that as stale starves the env forever.
    bridge, savegame = make_bridge(tmp_path)
    write_payload(savegame, seq=1_181_409)
    assert bridge.read_state(timeout=0.01) is not None

    write_payload(savegame, seq=2)
    state = bridge.read_state(timeout=0.01)
    assert state is not None
    assert state.seq == 2


def test_fresh_frames_advance(tmp_path):
    bridge, savegame = make_bridge(tmp_path)
    write_payload(savegame, seq=1)
    assert bridge.read_state(timeout=0.01).seq == 1
    write_payload(savegame, seq=2)
    assert bridge.read_state(timeout=0.01).seq == 2
