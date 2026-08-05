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

## BLOCKED (2026-08-05): Steam does not survive the 24.04 upgrade

The dist-upgrade REMOVED the steam package (`dpkg -l | grep steam` is empty,
`/usr/games/steam` is gone) while leaving the user-space install at
`~/.steam/steam/steam.sh`. That script bootstraps fine ("Steam runtime environment
up-to-date!") and stays alive in the foreground, but no client process persists when
detached, so nothing can launch the game.

**Diagnosed 2026-08-05. Do NOT try to apt-install the i386 stack. It is not survivable.**

Steam's own log (`~/.steam/steam/logs/console-linux.txt`) gives the real error, which the
apt output buries:

    Error: You are missing the following 32-bit libraries, and Steam may not run:
    libGL.so.1
    libdrm.so.2

The dist-upgrade dropped the i386 GL libraries. They cannot be put back:

- `libdrm2:i386` alone installs cleanly, so i386 multiarch itself is healthy.
- `libgl1:i386` fails outright; `libglx0:i386` appears to succeed, but only because apt
  satisfies it by REMOVING 146 packages - including `gdm3`, `gnome-shell`,
  `xserver-xorg-video-all`, `libegl-mesa0` and the whole `cuda-toolkit-12-2`. On a
  headless machine with no physical access that ends both the X session and the GPU
  work. Always check `apt-get install -s ... | grep '^Remv'` before running an i386
  install here; a resolver "success" is not a safe plan.

Two earlier theories were WRONG and should not be revived: the amd64 graphics stack is
healthy (libegl-mesa0/libglx0/mesa-libgallium all installed at matching versions), and
ESM is not amd64-only (`libqt5widgets5t64:i386` exists at the same `+esm1` version).
Nothing is in a broken dpkg state and nothing is held.

**Use Flatpak Steam.** It ships its own 32-bit libraries and touches none of the host
graphics stack, which is the only property that matters here:

```bash
sudo apt install flatpak && flatpak install -y flathub com.valvesoftware.Steam
```

Teardown is a Windows PE32+ binary run through Proton, so the Steam client stays on the
critical path - launching the exe directly is not an option. Budget a 4.4 GB re-download
into the sandbox. **The savegame bridge path moves** to
`~/.var/app/com.valvesoftware.Steam/.local/share/Steam/...`; `teardown_lab.savegame`
resolves it via `TEARDOWN_SAVEGAME`, so point that at the new location rather than
hardcoding it, and re-run the mod-reload check below.

Then re-verify: `uv run pytest -q`, then the strike check below.
Everything else survived the upgrade cleanly: CUDA (cu126) still works, the xorg
drop-in still forces HDMI-0, 92 tests pass. Driver is STILL 565.57.01 - the upgrade did
not cross 570 - so the Sunshine pin and the cu126 torch pin both stay as they are.

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
