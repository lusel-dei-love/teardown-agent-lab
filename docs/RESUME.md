<!-- ABOUTME: Where the project stands and what to do next, written to survive the -->
<!-- ABOUTME: Ubuntu 22.04 -> 24.04 upgrade and any interrupted session. -->

# Resume here

## State (2026-08-04)

Working:
- **Trained pixel student: 50-60% success vs 10% random.** Checkpoints in
  `runs/student_pitch/`. Observation is pixels + own proprioception only; no privileged
  state ever reaches the policy.
- **MolmoAct 2 (VLA) baseline runs**, bf16 on the 4090, 10.9 GB, ~1.3 s per decision.
- Harness: bridge, uinput actuator, supervisor, DAgger, fixed eval protocol. 90 tests.

## After the 24.04 upgrade, check these first

1. **X display number moves** (`ls /tmp/.X11-unix/`); everything resolves it at runtime
   via `teardown_lab.xdisplay.detect_display`, overridable with `GAME_DISPLAY`.
2. **Display output must be CONNECTED** or the game starts and never maps a window
   (`nvidia-smi --query-gpu=display_active`). The forcing config is
   `/etc/X11/xorg.conf.d/10-headless-nvidia.conf`; see `~/Setup/docs/headless-display-fix.md`.
3. **torch is pinned to cu126** in `pyproject.toml`. Driver >= 570 (which 24.04 brings)
   still runs cu126 fine - CUDA minor versions are forward compatible - so nothing needs
   changing. Only revisit if you want cu128+.
4. `~/42/FlowAI` `flowai-live.service` was **stopped and disabled** on Louis's say-so; it
   held 13.6 GB of VRAM. Re-enable with
   `systemctl --user enable --now flowai-live.service` when the GPU is free again.
5. Sunshine can be un-pinned once the driver is >= 570 (see the global CLAUDE.md).

## FIRST THING after the 24.04 upgrade

Strike attribution (lever 2) is committed but NOT verified live. The mod now credits a
block only if it moved while the agent was swinging within 3 m. Live before the last fix:
blind constant policy 40% -> 0/8 (intended), but the teacher also hit 0/8 because the mod
queried `InputDown("lmb")` while Teardown binds the sledge to `usetool`. The fix accepts
both names and is committed, unverified.

```bash
# reload the mod, then expect teacher ~85% and blind ~0
DISPLAY=:0 uv run python -c "from teardown_lab.real_bridge import RealBridge; print(RealBridge().hard_reset(timeout=120) is not None)"
```
If the teacher is still 0, the swing is not being seen at all: publish `InputDown` for
both names from the mod and check which one the actuator's uinput BTN_LEFT drives.

Once it passes: recollect (the strike rule changes what counts as success, so the old
datasets are not comparable), retrain, re-evaluate student vs blind vs random, and update
the dashboard.

## Next steps, in order

1. **Re-run both baselines with matched budgets.** Cosmos ran 3 episodes x 25 steps and
   MolmoAct 2 5 x 40, so their success rates are directional only. Cosmos needs
   transformers from git (`model_type cosmos3_edge` is absent from release 5.14.1 and the
   checkpoint ships no auto_map and no modeling code); it runs from a separate venv with
   `PYTHONPATH=<repo>/src`, and the game must already be in a level because that venv's
   newer python-xlib breaks pyautogui (only used for menu clicks).
2. **Calibrate the student's declare head on rollouts**, not the offline split. It is
   bimodal: at the tuned threshold it fires within ~16 steps and drops success to 10%; at
   0.995 with 4-frame hysteresis it never fires (60%). Sweep the threshold live.
3. **Rebuild the showcase** (deploy target lives in the gitignored `.envrc`, see
   `.envrc.example`) with the real comparison: random vs
   student stages vs MolmoAct 2 vs Cosmos, with videos per policy.

## Reproducing

```bash
uv sync --extra gui --extra train --extra baselines
uv run pytest -q                                     # 90 tests, no game needed
DISPLAY=:0 uv run python -m teardown_lab.collect --episodes 200 --out runs/x.npz
uv run python -m teardown_lab.train_student --dataset runs/x.npz --out-dir runs/y
DISPLAY=:0 uv run python -m teardown_lab.eval_student --stages runs/y
DISPLAY=:0 HF_HUB_OFFLINE=1 uv run python -m teardown_lab.baselines.run --model molmoact2
```

Long runs checkpoint and self-heal: collection saves every N episodes and restarts the
game on a crash (`teardown_lab.supervisor.ensure_playable`, dead -> playable in ~76 s).
