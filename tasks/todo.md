<!-- ABOUTME: Working plan for teardown-agent-lab; checkboxes mirror the milestone plan. -->
<!-- ABOUTME: See docs/superpowers/specs/2026-08-01-teardown-agent-lab-design.md for the spec. -->

# todo

## M6 — post-24.04 (2026-08-05)
- [x] upgrade survived: CUDA cu126 works, xorg drop-in still forces HDMI-0, 92 tests pass
- [x] driver still 565.57.01 (upgrade does NOT cross 570) -> Sunshine + torch pins stand
- [x] supervisor resolves the Steam launcher at runtime; the upgrade deleted
      /usr/games/steam and the hardcoded path crashed every launch
- [ ] **BLOCKED: Steam package removed by the dist-upgrade.** ~/.steam/steam/steam.sh
      bootstraps and survives in the foreground but no client persists when detached, so
      the game cannot be launched. i386 + 32-bit loader are present, so it is not the
      usual missing-lib cause. Needs `sudo apt install steam-installer` (Louis).
- [ ] then: verify strike attribution (teacher ~85%, blind ~0), recollect, retrain

## M5 — strike attribution (lever 2, 2026-08-04) — NEEDS LIVE VERIFICATION
- [x] mod credits a block only if it moved while swinging within 3 m; publishes a strike
      flag per block (payload now 11 fields). Referee requires displaced AND struck.
- [x] live: blind constant policy 40% -> 0/8, exactly the intended effect
- [ ] **BUT the teacher also went 0/8** - the mod queried InputDown("lmb") and Teardown
      binds the sledge to "usetool". Fix committed (accept either), NOT verified: the
      machine was rebooted for the 24.04 upgrade. Expect teacher ~85%, blind ~0.
- [ ] after it passes: recollect (old datasets score differently under the strike rule),
      retrain, re-evaluate, update the dashboard

## M4 — randomised tower bearing (2026-08-04)
- [x] mod places the tower on a seeded bearing (+/-180 deg) and distance (3.5-6.0 m)
      per episode; bearings measured live spanning -119 to +121 deg
- [x] collected 220 episodes (186 teacher successes, 85%), trained 20 epochs,
      val control_mse 0.0732
- [x] evaluated 10 episodes each: random 30% | constant forward+swing 40% |
      student 30%. **The student does NOT beat the controls on this variant.**
- [ ] **Open question: the randomised-BEARING task is still not discriminating.**
      A blind constant policy scores 40%, because with 220 steps (22 s) it wanders far
      enough to bump the tower, and the success rule (4 of 9 blocks moved >0.5 m) does
      not care who moved them or how. The variant that DID discriminate was spinning the
      camera before handover with the tower left ahead of spawn: constant 0/6 vs the
      easy-task student 2/6. Next levers:
      1. shorter episodes (the 22 s budget rewards wandering)
      2. success must require the agent to have STRUCK the tower, not merely disturbed it
      3. spawn the tower out of the walkable path so blind forward motion cannot reach it

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
## M1c — 200-episode run (2026-08-04)
- [x] display restored (xorg drop-in forces HDMI-0; 1600x900, no EDID so no 1080p mode)
- [x] collected 200 episodes / 23112 samples / 64% teacher success / 45.6 min, no crashes
- [x] declare positives 17 -> 1084 (4.7%) after widening the label to every solved frame
- [x] trained: val control MSE 0.123 -> 0.084, declare recall 0.81
- [x] evaluated live, 8 episodes per stage -- **STILL DOES NOT BEAT RANDOM**
      random 12.5% | stage_000 0% | stage_025 0% | stage_050 12.5% | stage_100 12.5%
      and with declare disabled entirely, stage_100 scores 0/8 -- so the CONTROL policy
      has not learned the task either; it is not merely a broken declare head.
## M1d — binary heads fixed, overfit check passed (2026-08-04)
- [x] grab/swing/declare now BCE classification heads, thresholded on probability
- [x] cosine LR annealing (a fixed 1e-3 floored the overfit loss at 3e-4)
- [x] datasets carry a per-sample episode index
- [x] **overfit sanity check PASSES**: 2 seeded episodes / 208 samples ->
      control_mse 1.42e-05, declare_bce 1.62e-05, total 3.18e-05, declare P=R=1.00.
      All 208 observations unique, zero conflicting labels. So the pipeline (labels,
      loss, normalisation, capacity) is sound; the residual is BCE's asymptote.
- [x] retrained on 200 episodes and evaluated live: the pathology is fixed but the
      policy is not. Live swing rate 74.6% -> 5.1% (teacher 0.25), success still 12.5%,
      i.e. random. It now UNDER-swings: a classifier trained on 25% positives that fires
      5% of the time on its own trajectories is predicting the majority class on frames
      it never saw.
## M1e — DAgger (2026-08-04)
- [x] DAgger loop: roll out student, label visited states with the teacher, aggregate,
      retrain, beta decay 0.50/0.25/0.12; 3 iterations x 25 episodes
- [x] every training metric improved monotonically: control_mse 0.0771 -> 0.0695,
      declare precision 0.25 -> 0.39 -> 0.52 (plain BC was 0.17)
- [x] live calibration fixed: swing fraction BC 0.207 -> DAgger 0.276 (teacher 0.25);
      mean episode length 18.8 -> 61.9 steps
- [x] BUT live success still 0/10, and 0/10 with the declare head disabled over full
      200-step episodes - so it is not a declare head gating a competent policy
## M3 — baselines (2026-08-04)
- [x] GPU unblocked: torch pinned to cu126 (default CUDA 13 wheels refuse driver 565)
- [x] shared text-action protocol: one prompt, one parser, one action vector, one
      actuator for VLA, world model and student alike
- [x] slow-mo (SetTimeScale) so a ~1.3 s/decision model is judged on decisions, not speed
- [x] **MolmoAct 2 (VLA): 0-20% success, 0% parse failures, but only 2 DISTINCT replies
      in 120 decisions** - a constant "walk forward and swing", not perception
- [x] **Cosmos 3 Edge (world model): 0.67 success (2/3), 0.88 unique replies, 6.3 s per
      decision.** Needs transformers from git; validated in a throwaway venv so the
      working MolmoAct 2 path stayed intact. It reasons about the frame explicitly then
      emits JSON.
- [x] **Head to head: the VLA is fluent and blind, the world model looks but is slow.**
      Reply diversity over 75/120 decisions is the solid signal (66 distinct vs 2); the
      success rates are directional only (n=3 and n=5, and different step budgets).
- [ ] re-run both with MATCHED episode counts and step budgets before publishing numbers
- [ ] (was) Cosmos 3 Edge BLOCKED model_type cosmos3_edge is absent from
      transformers 5.14.1 and the checkpoint ships no auto_map and no modeling code
      (MolmoAct 2 ships both). Needs transformers from git main - try it in a throwaway
      venv first, it can break the working MolmoAct 2 path.
- [ ] showcase page: random vs student stages vs MolmoAct 2 vs Cosmos, with videos

## M1f — WORKING POLICY (2026-08-04)
- [x] measured the real observation problem: target inside the 90 deg frustum in only
      20.4% of frames, because the teacher never pitched and the tower drops below the
      view as it closes in
- [x] teacher now aims pitch -> target on screen 0.675, teacher success 73-87%,
      episodes ~29 steps instead of ~165
- [x] frames raised to 224x126
- [x] tried and REJECTED: look_scale 200 -> 80. Measured worse (0.127 visible); finer
      steps just spend more frames mid-turn
- [x] recollected 187 episodes / 18676 samples, retrained: val control_mse 0.0583 (best
      ever; DAgger was 0.0695, BC 0.0842)
- [x] **live: 50-60% success vs 10% random.** Same declare-disabled test scored 0/10 for
      both the earlier BC and DAgger models
- [ ] **NEXT: calibrate the declare head on rollouts, not on the offline split.** It is
      bimodal - at the tuned threshold it fires within ~16 steps and ends episodes early
      (success drops to 10%); at 0.995 with 4-frame hysteresis it never fires. Sweep the
      threshold against live rollouts and pick the operating point that maximises
      success while keeping false declarations near zero.
- [ ] (resolved) perception, not optimisation or data volume The overfit check reached
      3.18e-05 on 2 episodes, so the model fits what it sees; held-out control_mse
      plateaus near 0.07, so it cannot predict the teacher's action from an unseen
      128x72 frame. The teacher reads exact 3D block positions; the student gets a
      heavily downsampled image of a 1600x900 screen. Try in order:
      1. higher-resolution or cropped frames (the tower occupies few pixels at 128x72)
      2. frame stacking (a single frame carries no motion)
      3. a larger visual encoder
- [ ] (superseded) compounding error / distribution shift
      Collection only approximates DAgger - 5% of samples come from random scrambles,
      and none from the STUDENT's own state distribution. Real DAgger: roll out the
      current student, label the states IT visits with the privileged teacher, append,
      retrain, repeat. Everything needed is already in place (teacher is queryable at
      any state, collection loop and checkpointing exist); it is a loop around them.
- [x] (resolved) binary actions were being trained as regression.
      The teacher's `swing` is binary and on for 25% of frames. MSE drives the student to
      output ~0.25 everywhere, and the env thresholds `swing = vec[5] > 0`, so the student
      swings on 74.6% of frames - flailing instead of approaching and striking.
      `look_dx` shows the same damping (teacher std 0.743 vs student 0.451).
      Fix: train `swing` and `grab` as BCE classification heads thresholded at p>0.5
      (exactly what already fixed the declare head), and consider discretising the
      continuous axes, since MSE on a multimodal action distribution regresses to the
      mean by construction.
- [ ] (resolved) BLOCKED ON HARDWARE: no display output is connected. Every xrandr output reads
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
