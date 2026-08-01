# ABOUTME: Tests for the Bridge protocol and the in-memory FakeBridge test double.
# ABOUTME: No transport, no game: pure sequencing/recording behaviour.

from teardown_lab.bridge import Bridge, FakeBridge

from tests.helpers import state_with_displaced_blocks


def _states(n: int) -> list:
    return [state_with_displaced_blocks(i, 1.0, t=float(i)) for i in range(n)]


def test_fake_bridge_satisfies_protocol():
    assert isinstance(FakeBridge(_states(1)), Bridge)


def test_reads_states_in_order():
    states = _states(3)
    bridge = FakeBridge(states)
    assert bridge.read_state() == states[0]
    assert bridge.read_state() == states[1]
    assert bridge.read_state() == states[2]


def test_repeats_last_state_when_exhausted():
    states = _states(2)
    bridge = FakeBridge(states)
    bridge.read_state()
    bridge.read_state()
    assert bridge.read_state() == states[-1]
    assert bridge.read_state() == states[-1]


def test_read_state_returns_none_without_states():
    assert FakeBridge([]).read_state() is None


def test_send_records_commands():
    bridge = FakeBridge(_states(2))
    bridge.send({"cmd": "noop"})
    bridge.send({"cmd": "reset", "seed": 7})
    assert bridge.sent == [{"cmd": "noop"}, {"cmd": "reset", "seed": 7}]


def test_reset_rewinds_to_first_state():
    states = _states(3)
    bridge = FakeBridge(states)
    bridge.read_state()
    bridge.read_state()
    bridge.send({"cmd": "reset", "seed": 3})
    assert bridge.read_state() == states[0]
    assert bridge.read_state() == states[1]


def test_non_reset_command_does_not_rewind():
    states = _states(3)
    bridge = FakeBridge(states)
    bridge.read_state()
    bridge.send({"cmd": "noop"})
    assert bridge.read_state() == states[1]


def test_close_is_recorded():
    bridge = FakeBridge(_states(1))
    assert bridge.closed is False
    bridge.close()
    assert bridge.closed is True
