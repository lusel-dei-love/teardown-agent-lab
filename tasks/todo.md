<!-- ABOUTME: Working plan for teardown-agent-lab; checkboxes mirror the milestone plan. -->
<!-- ABOUTME: See docs/superpowers/specs/2026-08-01-teardown-agent-lab-design.md for the spec. -->

# todo

## M0 — bridge
- [x] Verify Teardown modding API external-I/O surface -> registry/savegame.xml at >=49 Hz (see docs/superpowers/research/2026-08-01-bridge-transport.md)
- [x] Lua mod: 3x3 tagged tower via Spawn() voxbox XML, state via registry, reset via HasFile rising edge
- [x] bridge.py transport + FakeBridge (savegame.py parser, RealBridge polling) -- verified live: 60 Hz reads + reset handshake

## M1 — env round-trip
- [x] actuator.py (uinput backend; XTest proven inert) + window focus management
- [x] capture.py (screenshots + ffmpeg episode recording)
- [x] env.py Gymnasium env against FakeBridge (unit-tested) and against the live game
- [x] scripted policy topples tower end-to-end -- success=True in 23.6s, 6/9 blocks displaced
- [ ] record a real episode trace as a game-free referee regression fixture

## M2 — training + eval
- [ ] train_sac.py (SB3) with staged checkpoints (0/25/50/100%)
- [ ] eval.py fixed-seed protocol -> metrics.json + per-episode MP4
- [ ] eval all four stages

## M3 — VLA baselines
- [ ] baselines/protocol.py text-action interface (shared)
- [ ] baselines/molmoact2.py (VLM path; optional native-mode garbage probe)
- [ ] baselines/cosmos_edge.py (transformers bf16)
- [ ] run protocol for both, record latency/VRAM + slow-mo factor

## M4 — showcase
- [ ] showcase/build.py: artifacts -> static HTML (stages, curve, baseline table, honesty notes)
- [ ] deploy via .envrc-configured destination; "latest" section during training

## review
- [ ] (fill after each milestone)
