# ABOUTME: Shared test helpers for building synthetic GameState objects (a 9-block tower)
# ABOUTME: so referee/env tests never need the live game.

from teardown_lab.state import BlockState, GameState

# 3-wide, 3-high stack centred on the origin in XZ, stacked along +Y.
TOWER_SPAWNS: list[tuple[float, float, float]] = [
    (float(i % 3) - 1.0, float(i // 3), 0.0) for i in range(9)
]


def state_with_displaced_blocks(
    n_displaced: int,
    dist: float,
    *,
    t: float = 0.0,
    seed: int = 1,
    episode: int = 0,
    player_pos: tuple[float, float, float] = (0.0, 0.0, -5.0),
    yaw: float = 0.0,
    pitch: float = 0.0,
) -> GameState:
    """A 9-block tower where the first `n_displaced` blocks sit `dist` off their spawn."""
    blocks = []
    for i, spawn in enumerate(TOWER_SPAWNS):
        if i < n_displaced:
            pos = (spawn[0], spawn[1], spawn[2] + dist)
        else:
            pos = spawn
        # Displaced blocks in fixtures represent blocks the agent actually hit; the
        # referee now requires that attribution.
        blocks.append(BlockState(pos=pos, spawn=spawn, struck=i < n_displaced))
    return GameState(
        t=t,
        seed=seed,
        episode=episode,
        player_pos=player_pos,
        yaw=yaw,
        pitch=pitch,
        blocks=blocks,
    )
