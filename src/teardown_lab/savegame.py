# ABOUTME: Parses the mod's state payload out of Teardown's savegame.xml registry file.
# ABOUTME: Pure functions over text: no filesystem or game dependency, fully unit-tested.

from __future__ import annotations

from teardown_lab.state import BlockState, GameState

# The engine scopes a local mod's registry keys under savegame/mod/local-<modfolder>.
MOD_ELEMENT = "local-{mod}"


class PayloadError(ValueError):
    """The registry held a payload this parser cannot read."""


def _vec(text: str) -> tuple[float, float, float]:
    parts = text.split(",")
    if len(parts) != 3:
        raise PayloadError(f"expected 3 components, got {text!r}")
    x, y, z = (float(p) for p in parts)
    return (x, y, z)


def parse_payload(payload: str) -> GameState:
    """Decode the mod's pipe-delimited state payload.

    Format: seq|t|episode|seed|px,py,pz|yaw,pitch|cur_blocks|spawn_blocks
    where block groups are ';'-separated vectors. The format deliberately avoids XML
    metacharacters: the engine writes savegame.xml without entity escaping, so a JSON
    payload would corrupt the document.
    """
    fields = payload.split("|")
    # One format, no guessing. Earlier shapes (8 and 9 fields) were ambiguous once
    # `phase` was added, and the mod ships with this parser - a stale build should fail
    # loudly here rather than be silently misread.
    if len(fields) != 11:
        raise PayloadError(f"expected 11 fields, got {len(fields)}")

    seq, t, episode, seed, player, look, phase, cur, spawn, local, hit = fields

    yaw_s, _, pitch_s = look.partition(",")
    if not pitch_s:
        raise PayloadError(f"malformed look field {look!r}")

    cur_list = [c for c in cur.split(";") if c]
    spawn_list = [s for s in spawn.split(";") if s]
    if len(cur_list) != len(spawn_list):
        raise PayloadError(
            f"block count mismatch: {len(cur_list)} current vs {len(spawn_list)} spawn"
        )

    hit_list = [h for h in hit.split(";") if h]
    if hit_list and len(hit_list) != len(cur_list):
        raise PayloadError(
            f"strike-flag count mismatch: {len(hit_list)} vs {len(cur_list)} blocks"
        )
    if not hit_list:
        hit_list = ["0"] * len(cur_list)

    blocks = [
        BlockState(pos=_vec(c), spawn=_vec(s), struck=(h == "1"))
        for c, s, h in zip(cur_list, spawn_list, hit_list, strict=True)
    ]

    blocks_local = [_vec(v) for v in local.split(";") if v]

    return GameState(
        t=float(t),
        seed=int(seed),
        episode=int(episode),
        player_pos=_vec(player),
        yaw=float(yaw_s),
        pitch=float(pitch_s),
        blocks=blocks,
        seq=int(seq),
        blocks_local=blocks_local,
        phase=int(phase),
    )


def extract_payload(xml_text: str, mod: str) -> str | None:
    """Pull the raw state string out of savegame.xml, or None if the mod hasn't written.

    Deliberately a targeted scan rather than an XML parse. Two reasons: this runs at the
    control loop's rate against a file the engine rewrites whole every frame, so building
    a full element tree to read one attribute is wasteful; and not invoking an XML parser
    at all removes the entity-expansion attack surface. It is unambiguous because the
    payload is guaranteed free of XML metacharacters by the mod's format.

    Returns None (rather than raising) for a torn or truncated read: the engine rewrites
    this file via a temp file and rename, so a reader can catch it mid-write.
    """
    anchor = f"<{MOD_ELEMENT.format(mod=mod)}>"
    start = xml_text.find(anchor)
    if start < 0:
        return None

    marker = '<state value="'
    vstart = xml_text.find(marker, start)
    if vstart < 0:
        return None
    vstart += len(marker)

    vend = xml_text.find('"', vstart)
    if vend < 0:
        return None
    return xml_text[vstart:vend]
