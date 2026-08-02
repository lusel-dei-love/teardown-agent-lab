<!-- ABOUTME: Approved design spec for teardown-agent-lab: train/eval/showcase an RL agent -->
<!-- ABOUTME: in Teardown and benchmark MolmoAct 2 + Cosmos 3 Edge as zero-shot baselines. -->

# teardown-agent-lab — Design

**Status:** approved 2026-08-01; **amended 2026-08-02** (observation, termination,
training method — see "Amendment" below, which supersedes the original v1 text where
they conflict). **Scope:** v1 (single task, single machine).

## Amendment 2026-08-02 — no privileged information in the policy's input

The original v1 fed the agent privileged state (player pose + every block pose). That is
now disallowed. The rule:

> **Privileged state may be used in the loss, the reward, the teacher, and the eval
> scoring. It must never enter the policy's observation.**

Three consequences:

1. **Observation = pixels + own proprioception.** The policy sees a downsampled RGB frame
   of the game window plus its OWN yaw, pitch and velocity. The line is *self vs world*:
   a real robot has joint encoders, so self-state is fair; anything about the world
   (block poses, tower geometry) must be inferred from pixels. Measured cost: 8 ms to
   grab 1920x1080, 21 ms to downsample to 128x72 - 29 ms of the 100 ms step budget.

2. **Termination is no longer an oracle.** Ending the episode the instant the privileged
   referee fires leaks privileged information through the episode structure, and means
   the agent never has to *look* at what it did. Instead the action space gains a
   **declare-complete** action: the agent decides the task is done, which ends the
   episode, and we score whether it was right. This makes visual verification
   instrumentally necessary and yields a metric the old design could not express -
   **false-declaration rate** (declared complete while the tower still stands).
   Episodes still truncate on the timeout.

3. **Training is teacher -> student distillation.** The privileged scripted policy (which
   already solves the task) acts as teacher; the pixel policy learns to imitate it,
   DAgger-style, with privileged state present only in the loss. Chosen over end-to-end
   asymmetric RL because the game caps us at ~36k environment steps per hour (no headless
   mode, no parallel instances), where pure pixel RL would plausibly need 100k-500k steps
   per run. Asymmetric actor-critic finetuning stays open as a follow-up.

**This also fixes the benchmark.** The RL agent and the MolmoAct 2 / Cosmos 3 Edge
baselines now share an observation space (pixels) and an action interface (uinput), so
the comparison is like-for-like rather than privileged-vs-pixels.

## Goal

A minimal, reproducible lab that (1) trains an RL agent on a task inside the voxel
destruction game Teardown, (2) evaluates every policy with one fixed protocol, and
(3) publishes a visual showcase of the policy at different training stages, alongside
zero-shot baselines from two open vision-language-action / world-foundation models:
MolmoAct 2 (`allenai/MolmoAct2` family) and NVIDIA Cosmos 3 Edge (`nvidia/Cosmos3-Edge`).

The interesting claim v1 can support: a small task-specific RL policy trained on privileged
state beats (or does not beat) frontier generalist models driven zero-shot through a fair
text-action interface, on a task none of them has seen.

## v1 task: tower knockdown

A custom Teardown sandbox level containing a tower of loose blocks. Episode: agent spawns
at a fixed pose, timeout T seconds. Success: the tower's maximum block height drops below
a threshold (equivalently >= K blocks displaced from their spawn poses). Reward (shaped,
computed host-side in referee.py): approach term + per-step displacement delta + success bonus.
Defaults (tunable constants, recalibrated at M1 against the real level): T = 60 s,
tower = 3x1 stack of 9 blocks, K = 5, displacement threshold = 0.5 m from spawn pose.

## Architecture

```
Teardown (Proton, X display, window class steam_app_1167630)
  └─ Lua mod  = sensor + reset only: streams body/player transforms, handles
     deterministic reset (Restart + SetRandomSeed). Reward/success are computed
     HOST-SIDE in Python (referee.py, pure functions, unit-tested) from that state.
bridge.py     = transport abstraction to the mod. Preference order:
                 (1) native HTTP if the modding API exposes it  [verify FIRST]
                 (2) file mailbox in the mod's writable dir
                 (3) log-tail out + input-injection in
actuator.py   = OS-level input injection into the game window (pyautogui/python-xlib;
                 xdotool optional). One actuator for ALL policies - RL and VLA baselines
                 play through the same interface a human uses.
capture.py    = screenshots (VLA observations) + ffmpeg episode recording (X11 grab)
env.py        = Gymnasium env: obs = compact state vector (tower body poses, player pose,
                 camera), action = look delta (2D) + move + grab/swing buttons, 10 Hz
train_sac.py  = stable-baselines3 SAC, single env instance, checkpoints at fixed
                 fractions of training (0/25/50/100% = the showcase stages)
eval.py       = ONE protocol for every policy: 20 episodes, fixed seed list, fixed
                 timeout -> metrics.json (success rate, time-to-topple) + per-episode MP4
baselines/    = text-action protocol shared by both VLAs: screenshot + instruction +
                 action-vocabulary description -> model emits structured action ->
                 actuator executes. Latency compensated with in-game slow-mo
                 (SetTimeScale), reported explicitly as an asterisk.
showcase/     = build.py renders eval artifacts into one static self-contained HTML page
                 (stage progression videos, learning curve, baseline table); deploy via
                 scp to an OAuth-gated host configured ONLY via gitignored .envrc.
                 During training, a "latest" section (most recent episode + curve) is
                 pushed on a timer.
```

### Why acting is OS-level injection, not Lua

Acting via Lua (`SetPlayerTransform` etc.) is deterministic but teleport-y and would make
the VLA comparison apples-to-oranges. Injection means every policy plays the game the way
a human does; Lua stays the sensor/referee. Lua-side acting remains a documented fallback
for primitives injection cannot do reliably.

### Baselines: why text-mediated

Neither model emits game actions natively. MolmoAct 2's `predict_action` requires a robot
joint-state vector and a closed set of normalization tags; Cosmos 3 Edge's action heads
are locked to robot/AV embodiments. The fair, runnable zero-shot baseline drives each
model's VLM half (screenshot + instruction -> structured text action) through the shared
protocol. Optionally, one documented "native mode" probe run (fake state vector) records
that the robot action head emits garbage on game input - a datapoint, not a benchmark.

## Training approach

SAC on the privileged state vector (off-policy = right choice for one non-headless,
real-time game instance; no teleop rig needed). Decision rate 10 Hz. Reward pure
functions are unit-tested host-side against recorded state traces.

## Evaluation & honesty rules

- Same seeds, same timeout, same success predicate for every policy.
- Report success rate over >= 20 episodes; multiple eval runs if variance is visible.
- Any clock manipulation (slow-mo for VLA latency) is reported next to the number.
- The RL agent uses privileged state; the VLAs use pixels. This asymmetry is stated
  prominently in the showcase - v1 measures "task-specific RL vs zero-shot generalist",
  not matched-observation ablations.

## Constraints (non-negotiable)

- **Identity guard:** this public repo must never contain personal identity tokens,
  infra hostnames, or the deploy target. Deploy config lives in gitignored `.envrc`
  (committed `.envrc.example` has empty values). The pre-commit guard enforces this;
  a blocked commit is the guard working.
- **No game assets in the repo.** The Lua mod is original code; eval videos are our own
  gameplay captures. Internal R&D use per the retail EULA; no game-derived data resale.
- House rules: uv (never bare python), ABOUTME headers, rip not rm, no --no-verify.

## Risks

1. **Bridge transport unknown** - the modding API may or may not expose HTTP; verified
   as the FIRST implementation task, with the two fallbacks designed in.
2. **Single-instance sample throughput** caps RL performance; acceptable - the showcase
   is about progression, not SOTA.
3. **VLA latency** (~0.2-1 Hz decisions) - mitigated by slow-mo + generous timeout.
4. Cosmos 3 Edge is untested on Ada GPUs (bf16-only) - likely fine, we are early.
5. Game/Proton flakiness - the harness must survive game restarts (bridge reconnect,
   episode invalidation on crash).

## Milestones

- **M0** bridge transport verified; mod skeleton returns live state to Python
- **M1** Gymnasium env + actuator round-trip on the real game; scripted policy topples
  the tower (validates reward + success predicate end-to-end)
- **M2** SAC training run with staged checkpoints; eval protocol produces metrics + videos
- **M3** both VLA baselines run the protocol zero-shot
- **M4** showcase page live (stages + curves + baseline table + honesty notes)
