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

## BLOCKED: flatpak 1.14.6 is too old for Steam (2026-08-07)

Everything else is healthy: driver **580.178.04**, torch sees CUDA, the matching
`nvidia-580-178-04` GL/GL32 flatpak extensions are installed, display `:0`, session
unlocks with `~/Setup/scripts/unlock-session.sh`.

`kernel.apparmor_restrict_unprivileged_userns=0` is now set and persisted in
`/etc/sysctl.d/60-flatpak-userns.conf`. That fixed userns ON THE HOST - a plain
`unshare --user` succeeds. It is NOT sufficient, and this is the distinction that cost a
lot of time:

```
# host           -> works
unshare --user --map-root-user true
# inside sandbox -> "unshare failed: Operation not permitted"
flatpak run --command=sh com.valvesoftware.Steam -c 'unshare --user --map-root-user true'
```

Steam needs a NESTED user namespace inside the flatpak sandbox (pressure-vessel).
Flatpak 1.14.6 - what noble ships, and noble-backports has nothing newer - blocks the
`unshare` syscall in its seccomp filter. Nested userns support landed in the flatpak
1.15 series. `features=devel` is already granted, so no permission toggle helps.

Downgrading the Steam flatpak does NOT work - tried 1.0.0.84 (2025-10-02, the oldest
build flathub still lists) and it fails identically. The app has been restored to latest
and unmasked.

**Needs root (Louis):**

```bash
sudo add-apt-repository -y ppa:flatpak/stable && sudo apt update && sudo apt install -y flatpak
```

Then verify `flatpak --version` >= 1.15, re-check the nested-userns one-liner above, and
resume at the Flatpak section. The mod still needs enabling once in Play > Mod manager,
where it now appears under "Local files".

If the PPA is unwanted, the alternative is dropping Flatpak entirely and finding another
way to supply the i386 GL libraries the packaged Steam needs - but note apt cannot
install those without removing 146 packages (see below).

## Steam runs via FLATPAK now (2026-08-05)

The 24.04 upgrade removed the steam package and dropped the i386 GL libraries
(`libGL.so.1`, `libdrm.so.2` - see `~/.steam/debian-installation/logs/console-linux.txt`).
They cannot be restored: `libgl1:i386` fails, and `libglx0:i386` only "resolves" because
apt satisfies it by REMOVING 146 packages including `gdm3`, `gnome-shell`,
`xserver-xorg-video-all` and `cuda-toolkit-12-2`. Always
`apt-get install -s <pkg> | grep '^Remv'` before an i386 install here. Two theories
recorded earlier were WRONG: the amd64 stack is healthy, and ESM is not amd64-only
(`libqt5widgets5t64:i386` exists at the same `+esm1` version).

Flatpak Steam ships its own 32-bit stack (`Compat.i386`, `GL32.nvidia-565-57-01` matches
the host driver exactly) and touches no host package. Installed at USER scope - the
`sudo apt install flatpak && flatpak install ...` one-liner fails with "Flatpak system
operation Deploy not allowed for user" because sudo only covers the apt half:

```bash
flatpak remote-add --user --if-not-exists flathub https://dl.flathub.org/repo/flathub.flatpakrepo
flatpak install --user -y flathub com.valvesoftware.Steam
```

It reuses the EXISTING 13 GB library, the Steam login and the installed mod - no
re-download - via a symlink plus a filesystem grant:

```bash
flatpak override --user --filesystem="$HOME/.steam" com.valvesoftware.Steam
ln -s "$HOME/.steam/debian-installation" ~/.var/app/com.valvesoftware.Steam/.local/share/Steam
```

Do NOT also symlink `~/.var/app/com.valvesoftware.Steam/.steam`: the sandbox remaps
`$HOME` onto the app dir, so that symlink points at itself and bwrap fails to
bind-mount the real library.

Launch (the client now SURVIVES detaching, which the packaged one did not):

```bash
setsid nohup flatpak run --env=DISPLAY=:0 com.valvesoftware.Steam -silent &
setsid nohup flatpak run --env=DISPLAY=:0 com.valvesoftware.Steam steam://rungameid/1167630 &
```

`RealBridge` resolves the prefix at runtime (`default_prefix()`, `TEARDOWN_PREFIX`), so
both Steam layouts work unchanged.

**UNVERIFIED / next step:** Steam starts the game (reaper + pressure-vessel alive,
console log adds processes for gameID 1167630) but **no X window had mapped after ~3
min** and `wmctrl -l` listed nothing on `:0`. Check whether the session is still on `:0`
(`ls /tmp/.X11-unix/`), whether the output is CONNECTED
(`nvidia-smi --query-gpu=display_active --format=csv`), and whether this is just a slow
first Proton run under the new flatpak runtime. Do not assume the flatpak move caused it -
the "starts but never maps a window" failure predates it and is documented above.

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
