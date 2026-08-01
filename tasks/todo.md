<!-- ABOUTME: Working plan for teardown-agent-lab; checkboxes mirror the milestone plan. -->
<!-- ABOUTME: See docs/superpowers/specs/2026-08-01-teardown-agent-lab-design.md for the spec. -->

# todo

## M0 — bridge
- [ ] Verify Teardown modding API external-I/O surface (HTTP? file I/O? registry?) against current docs
- [ ] Lua mod skeleton: spawn-tagged tower level, state read, reward/success, Restart/SetRandomSeed reset
- [ ] bridge.py with chosen transport + FakeBridge for host-side tests

## M1 — env round-trip
- [ ] actuator.py (pyautogui/python-xlib) + window focus management
- [ ] capture.py (screenshots + ffmpeg episode recording)
- [ ] env.py Gymnasium env against FakeBridge (unit-tested), then real game
- [ ] scripted policy topples tower end-to-end (validates reward + success)

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
