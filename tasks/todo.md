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

## M1b — non-privileged agent (amendment 2026-08-02)
- [x] frames.py: downsampled RGB grabber (16 ms/frame after stride optimisation)
- [x] pixel_env.py: obs = pixels + own proprioception only; declare action ends episode
- [x] teacher.py: privileged expert emitting student-shaped actions incl. declare
- [x] verified live: teacher solves + declares correctly at a true 10.0 Hz
- [x] collect teacher demonstration dataset (4097 samples / 30 episodes, 17 successes)
- [x] train pixel student (CNN + proprio head); val control MSE 0.093
- [x] eval student vs random vs untrained -- NEGATIVE RESULT, see below
- [ ] **student does not solve the task yet.** stage_100 declares falsely at step ~12 in
      4/4 episodes; every earlier stage never declares and times out. Ranked fixes:
      1. more data: 30 episodes is tiny for pixel BC. Collection runs at ~4 episodes/min,
         so 200+ episodes is <1 h -- the cheapest lever by far.
      2. declare labels: only 17 positives in 4097 samples. Label the whole post-success
         verification window as declare-positive (~8x more) instead of the single step.
      3. teacher ceiling: ~57% success means even perfect imitation caps there.
      4. no motion cue in the observation: single frames + proprio velocity. Stack 2-4
         frames so the student can see blocks falling.
- [x] supervisor: dead game -> playable level in 76 s (process, Steam, workspace, menus)
- [x] runtime X display detection (number moves across reboots)
- [x] find the largest MAPPED game window; freshness-checked in_level
- [x] menu targets derived from live window geometry (absolute pixels died at 4K)
- [x] never click the character-select coordinate: it is Quit on the main menu
- [ ] **BLOCKED ON HARDWARE: no display output is connected.** Every xrandr output reads
      `disconnected`, `nvidia-smi` shows display_active Disabled, and Teardown logs
      `Display resolution: 0x0` then never maps a window. Needs one of:
      (a) the monitor powered on / cable reseated (no physical access),
      (b) a dummy HDMI/DP plug, or
      (c) a forced virtual output - xorg.conf `AllowEmptyInitialConfiguration` +
          `ConnectedMonitor`, which needs sudo.
      Option (c) is the durable fix for an unattended RL box and would also stop the
      Sunshine bridge breaking the same way. A 200-episode attempt
      reached episode 41 (24 successes, 59%) before Teardown hard-crashed; Steam then
      refused to relaunch it. Collection now checkpoints, so a crash costs one chunk
      rather than everything, but unattended runs need: detect process death -> relaunch
      Steam+game -> drive back into a sandbox level -> resume. Until then, collect in
      chunks of <=60 episodes with the game restarted between them.

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
