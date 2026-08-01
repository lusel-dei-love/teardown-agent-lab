# ABOUTME: Tests for parsing the mod's state payload out of Teardown's savegame.xml.
# ABOUTME: Uses captured real-shape fixtures; no game or filesystem required.

import pytest

from teardown_lab.savegame import PayloadError, extract_payload, parse_payload

PAYLOAD = (
    "42|1.500|3|3|1.000,2.000,3.000|90.000,-5.000|"
    "0.100,0.200,0.300;1.100,1.200,1.300|"
    "0.000,0.000,0.000;1.000,1.000,1.000"
)

# Mirrors the real file: the mod's keys are scoped under local-<modfolder>.
SAVEGAME = f"""<?xml version="1.0" encoding="UTF-8"?>
<registry>
	<savegame>
		<mod>
			<local-teardownlab>
				<ready value="1"/>
				<state value="{PAYLOAD}"/>
			</local-teardownlab>
		</mod>
	</savegame>
</registry>
"""


def test_extract_payload_finds_the_mod_scoped_value():
    assert extract_payload(SAVEGAME, "teardownlab") == PAYLOAD


def test_extract_payload_returns_none_for_a_different_mod():
    assert extract_payload(SAVEGAME, "othermod") is None


def test_extract_payload_returns_none_on_a_torn_read():
    # The engine rewrites the file whole; a reader can catch it mid-write.
    truncated = SAVEGAME[: SAVEGAME.index("<state value=") + 20]
    assert extract_payload(truncated, "teardownlab") is None


def test_parse_payload_round_trips_fields():
    state = parse_payload(PAYLOAD)
    assert state.seq == 42
    assert state.t == pytest.approx(1.5)
    assert state.episode == 3
    assert state.seed == 3
    assert state.player_pos == pytest.approx((1.0, 2.0, 3.0))
    assert state.yaw == pytest.approx(90.0)
    assert state.pitch == pytest.approx(-5.0)
    assert len(state.blocks) == 2
    assert state.blocks[0].pos == pytest.approx((0.1, 0.2, 0.3))
    assert state.blocks[0].spawn == pytest.approx((0.0, 0.0, 0.0))


def test_parse_payload_rejects_block_count_mismatch():
    bad = PAYLOAD.rsplit("|", 1)[0] + "|0.000,0.000,0.000"
    with pytest.raises(PayloadError, match="mismatch"):
        parse_payload(bad)


def test_parse_payload_rejects_wrong_field_count():
    with pytest.raises(PayloadError, match="8 or 9 fields"):
        parse_payload("1|2|3")


def test_parse_payload_reads_camera_space_blocks():
    payload = PAYLOAD + "|0.500,0.000,-2.000;0.600,0.000,-2.100"
    state = parse_payload(payload)
    assert len(state.blocks_local) == 2
    # Camera space: +x right, -z forward, so the tower sits ahead and slightly right.
    assert state.blocks_local[0] == pytest.approx((0.5, 0.0, -2.0))


def test_parse_payload_without_camera_space_still_parses():
    # An older mod build emits 8 fields; the host must not hard-fail on it.
    state = parse_payload(PAYLOAD)
    assert state.blocks_local == []


def test_payload_contains_no_xml_metacharacters():
    # The engine writes savegame.xml without entity escaping, so any of these in the
    # payload would corrupt the document for every reader.
    assert not set(PAYLOAD) & set('<>&"')
