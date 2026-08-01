# teardown-agent-lab

A minimal lab for training, evaluating, and showcasing agents that play
[Teardown](https://teardowngame.com/) — plus zero-shot baselines from open
generalist models (MolmoAct 2, NVIDIA Cosmos 3 Edge) driven through the same
action interface.

**v1 task:** tower knockdown. An RL agent (SAC on privileged game state, read by a
Lua mod) learns to topple a block tower; every policy — including the VLA baselines —
acts through OS-level input injection and is scored by one fixed evaluation protocol
(fixed seeds, success rate over 20 episodes, per-episode video).

## Status

Early development. See `docs/superpowers/specs/` for the design and
`docs/superpowers/plans/` for the implementation plan.

## Layout

- `mod/` — Teardown Lua mod: state sensor, reward/success referee, deterministic reset
- `src/teardown_lab/` — bridge, Gymnasium env, actuator, capture, training, eval,
  VLA baselines, showcase builder
- `tasks/` — working notes (todo, lessons)

## Honesty notes

The RL agent uses privileged state; the VLA baselines use pixels through a
text-action protocol (their robot-embodiment action heads cannot emit game inputs).
Any in-game clock manipulation used to compensate model latency is reported next to
the resulting numbers.

No game assets are redistributed here. Gameplay captures are our own; usage is
internal R&D within the retail EULA.

## License

MIT (code only).
