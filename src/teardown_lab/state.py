# ABOUTME: Plain dataclasses for the privileged game state streamed by the Lua mod
# ABOUTME: (player pose + tagged block poses) plus its JSON wire (de)serialization.

from __future__ import annotations

import json
from dataclasses import dataclass

Vec3 = tuple[float, float, float]


def _vec3(raw) -> Vec3:
    x, y, z = raw
    return (float(x), float(y), float(z))


@dataclass(frozen=True)
class BlockState:
    """One tagged dynamic block: where it is now, and where it spawned."""

    pos: Vec3
    spawn: Vec3

    def to_dict(self) -> dict:
        return {"pos": list(self.pos), "spawn": list(self.spawn)}

    @classmethod
    def from_dict(cls, payload: dict) -> BlockState:
        return cls(pos=_vec3(payload["pos"]), spawn=_vec3(payload["spawn"]))


@dataclass(frozen=True)
class GameState:
    """A single tick of privileged state: episode bookkeeping, player pose, blocks."""

    t: float
    seed: int
    episode: int
    player_pos: Vec3
    yaw: float
    pitch: float
    blocks: list[BlockState]
    # Monotonic tick counter from the mod. Lets a reader tell a fresh frame from a
    # re-read of the same one, since the transport is a polled file.
    seq: int = 0

    def to_dict(self) -> dict:
        return {
            "t": self.t,
            "seed": self.seed,
            "episode": self.episode,
            "player_pos": list(self.player_pos),
            "yaw": self.yaw,
            "pitch": self.pitch,
            "blocks": [b.to_dict() for b in self.blocks],
            "seq": self.seq,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, payload: dict) -> GameState:
        return cls(
            t=float(payload["t"]),
            seed=int(payload["seed"]),
            episode=int(payload["episode"]),
            player_pos=_vec3(payload["player_pos"]),
            yaw=float(payload["yaw"]),
            pitch=float(payload["pitch"]),
            blocks=[BlockState.from_dict(b) for b in payload["blocks"]],
            seq=int(payload.get("seq", 0)),
        )

    @classmethod
    def from_json(cls, s: str) -> GameState:
        return cls.from_dict(json.loads(s))
