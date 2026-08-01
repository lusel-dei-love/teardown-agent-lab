<!-- ABOUTME: Implementation plan for teardown-agent-lab milestones M0-M2: bridge, env, -->
<!-- ABOUTME: training + eval. M3 (VLA baselines) and M4 (showcase) get a later plan. -->

# Core Harness (M0-M2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A working train/eval loop for tower knockdown in Teardown: Lua mod streams state, Python computes reward, SAC trains through OS input injection, one fixed eval protocol emits metrics + videos.

**Architecture:** Lua mod = sensor + reset only. Python owns everything else behind narrow interfaces: `Bridge` (transport), `Actuator` (input injection with injectable backend), `referee` (pure reward/success), `TeardownTowerEnv` (Gymnasium), `eval` (fixed-seed protocol). Every unit is testable against fakes; the live game is only needed for Tasks 1, 8, 9 and real runs.

**Tech Stack:** Python 3.12 via uv; gymnasium, stable-baselines3, numpy, pytest; pyautogui + python-xlib + mss + pillow (gui extra); ffmpeg (system); Teardown 2.0.4 under Proton on X display `:1`, window class `steam_app_1167630`.

## Global Constraints

- `uv run` for every entry point; never bare python.
- Every code file starts with a 2-line `ABOUTME:` comment.
- Identity guard: no personal tokens, hostnames, or deploy targets in any file; deploy config only via gitignored `.envrc`. A blocked commit = guard working; fix content, never bypass.
- No game assets committed; mod code is original.
- `rip` for deletion, never `rm`. Commit after every green task; push only on request.
- Tests must not require the game unless marked `@pytest.mark.game` (excluded by default addopts).

## File Structure

```
pyproject.toml                      # uv project; extras: gui
mod/                                # Teardown mod (installed via symlink into the game's mod dir)
  info.txt  main.lua  ...           # exact layout fixed by Task 1's decision record
src/teardown_lab/
  __init__.py
  state.py          # GameState/BlockState dataclasses + JSON (de)serialization
  referee.py        # pure reward/success functions
  bridge.py         # Bridge protocol, FakeBridge, real transport (Task 8)
  actuator.py       # Action dataclass, Actuator with injectable backend, X11 backend
  env.py            # TeardownTowerEnv (Gymnasium)
  capture.py        # screenshot + ffmpeg episode recorder
  train_sac.py      # SB3 SAC + staged checkpoints (0/25/50/100%)
  eval.py           # fixed-seed protocol -> metrics.json (+ videos when live)
tests/              # mirrors src; unit tests run without the game
docs/superpowers/research/2026-08-01-bridge-transport.md   # Task 1 output
```

---

### Task 1: Bridge transport decision (research, game-side)

**Files:**
- Create: `docs/superpowers/research/2026-08-01-bridge-transport.md`

**Interfaces:**
- Produces: the chosen state-out / command-in mechanism, exact API calls, and update rate limits. Tasks 8-9 implement against this record. Everything in Tasks 2-7 is transport-agnostic and does NOT wait for this.

- [ ] **Step 1: Read the current official modding docs** (`https://teardowngame.com/modding/` and the API reference page it links, at game version 2.0.4). Enumerate every capability for getting data OUT of a running mod (HTTP/network functions, file write, registry/savegame persistence, clipboard, debug log path + format) and IN (file read, registry, launch params, HTTP responses).
- [ ] **Step 2: Confirm findings in-game** with a probe mod (temporary, not committed) that exercises the top candidate each way for 100 ticks; verify on the host that the data actually lands (file mtime / HTTP hits / log lines) and measure achievable rate.
- [ ] **Step 3: Write the decision record**: chosen transport out + in, exact Lua calls, serialization format, measured max Hz, fallback ranking, and the mod-install path for Proton (`<steam>/steamapps/compatdata/1167630/pfx/drive_c/users/steamuser/Documents/Teardown/mods` or as discovered).
- [ ] **Step 4: Commit** the decision record.

### Task 2: Project skeleton

**Files:**
- Create: `pyproject.toml`, `src/teardown_lab/__init__.py`, `tests/__init__.py`, `tests/test_smoke.py`

**Interfaces:**
- Produces: importable `teardown_lab` package; `uv run pytest` green; `gui` extra for X-dependent deps.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "teardown-agent-lab"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["gymnasium>=1.0", "numpy>=1.26", "stable-baselines3>=2.3"]

[project.optional-dependencies]
gui = ["pyautogui>=0.9", "python-xlib>=0.33", "mss>=9", "pillow>=10"]

[dependency-groups]
dev = ["pytest>=8"]

[tool.pytest.ini_options]
addopts = "-m 'not game'"
markers = ["game: requires the live game on an X display"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/teardown_lab"]
```

- [ ] **Step 2: Create the package** — `src/teardown_lab/__init__.py` with ABOUTME + `__version__ = "0.1.0"`; `tests/test_smoke.py` asserting `import teardown_lab` works.
- [ ] **Step 3: Run** `uv sync --all-extras && uv run pytest -q` — expect 1 passed.
- [ ] **Step 4: Commit.**

### Task 3: State model + referee (pure core)

**Files:**
- Create: `src/teardown_lab/state.py`, `src/teardown_lab/referee.py`
- Test: `tests/test_state.py`, `tests/test_referee.py`

**Interfaces:**
- Produces:
  - `BlockState(pos: tuple[float,float,float], spawn: tuple[float,float,float])`
  - `GameState(t: float, seed: int, episode: int, player_pos: tuple, yaw: float, pitch: float, blocks: list[BlockState])`; `GameState.from_json(s: str) -> GameState`; `.to_json()`
  - `referee.displacements(state) -> np.ndarray` (per-block XZ+Y distance from spawn)
  - `referee.success(state, k: int = 5, threshold: float = 0.5) -> bool`
  - `referee.reward(prev: GameState, curr: GameState, cfg: RewardConfig) -> float` with `RewardConfig(approach_w=0.05, displace_w=1.0, success_bonus=10.0, k=5, threshold=0.5)`

- [ ] **Step 1: Write failing tests** covering: round-trip JSON; displacement of a moved block; success exactly at k blocks past threshold (and not at k-1); reward = displace_w * (sum displacement delta) + approach_w * (prev_dist - curr_dist to tower centroid) + bonus only on the step success first becomes true.

```python
def test_success_requires_k_blocks():
    s = state_with_displaced_blocks(n_displaced=4, dist=1.0)
    assert not referee.success(s, k=5, threshold=0.5)
    s = state_with_displaced_blocks(n_displaced=5, dist=1.0)
    assert referee.success(s, k=5, threshold=0.5)

def test_reward_displacement_delta():
    prev = state_with_displaced_blocks(0, 0.0)
    curr = state_with_displaced_blocks(1, 0.4)   # below threshold still rewards delta
    cfg = RewardConfig(approach_w=0.0)
    assert referee.reward(prev, curr, cfg) == pytest.approx(0.4)
```

(`state_with_displaced_blocks` is a test helper building a 9-block GameState.)

- [ ] **Step 2: Run tests** — expect failures (module missing).
- [ ] **Step 3: Implement `state.py` + `referee.py`** as plain dataclasses + numpy; no I/O, no game imports.
- [ ] **Step 4: Run tests** — green. **Step 5: Commit.**

### Task 4: Bridge protocol + FakeBridge

**Files:**
- Create: `src/teardown_lab/bridge.py`
- Test: `tests/test_bridge.py`

**Interfaces:**
- Produces:
  - `class Bridge(Protocol): def read_state(self, timeout: float = 1.0) -> GameState | None; def send(self, cmd: dict) -> None; def close(self) -> None`
  - `FakeBridge(states: list[GameState])` — returns states in order, then repeats the last; records every `send()` in `.sent`; `send({"cmd":"reset","seed":n})` rewinds to the first state.

- [ ] **Step 1: Failing tests**: sequential reads; repeat-last; reset rewinds; sent commands recorded.
- [ ] **Step 2-4: Red, implement, green.** **Step 5: Commit.**

### Task 5: Actuator with injectable backend

**Files:**
- Create: `src/teardown_lab/actuator.py`
- Test: `tests/test_actuator.py`

**Interfaces:**
- Produces:
  - `Action(look_dx: float, look_dy: float, move_x: float, move_y: float, grab: bool, swing: bool)` (all in [-1,1]/bool)
  - `class InputBackend(Protocol): def move_mouse(self, dx: int, dy: int); def key(self, name: str, down: bool); def button(self, name: str, down: bool)`
  - `RecordingBackend` (test double, records calls)
  - `Actuator(backend, look_scale: int = 200)` with `.apply(action)` mapping: look -> `move_mouse(dx*look_scale, dy*look_scale)`; move_y>0.3 -> hold `w` (release when <=0.3), move_y<-0.3 -> `s`; move_x likewise `d`/`a`; grab -> right button hold; swing -> left button hold; `.release_all()`.
  - `X11Backend(display: str)` — thin pyautogui wrapper (no unit tests; exercised in game tasks); plus `find_game_window(display) -> int | None` matching WM_CLASS `steam_app_1167630`.

- [ ] **Step 1: Failing tests** for the mapping incl. hold/release hysteresis and `release_all` releasing exactly the held set.
- [ ] **Step 2-4: Red, implement, green.** **Step 5: Commit.**

### Task 6: Gymnasium env

**Files:**
- Create: `src/teardown_lab/env.py`
- Test: `tests/test_env.py`

**Interfaces:**
- Consumes: `Bridge`, `Actuator`, `referee`, `GameState`.
- Produces: `TeardownTowerEnv(bridge, actuator, cfg: EnvConfig, sleeper=time.sleep)` — Gymnasium `Env` with:
  - `EnvConfig(hz=10.0, timeout_s=60.0, k=5, threshold=0.5, n_blocks=9)`
  - observation: `Box(shape=(32,))` = player_pos(3) + yaw,pitch(2) + 9 blocks x relative pos(27)
  - action: `Box(low=-1, high=1, shape=(6,))` mapped to `Action` (grab/swing thresholded at 0)
  - `reset(seed=...)` sends `{"cmd":"reset","seed":seed}` to the bridge, reads first state
  - `step()` applies action, sleeps to the 10 Hz grid via injected `sleeper`, reads state, returns `(obs, reward, terminated, truncated, info)`; terminated on success, truncated on timeout; `info["success"]`, `info["t"]`.

- [ ] **Step 1: Failing tests** with `FakeBridge` + `RecordingBackend` + fake sleeper: obs shape/content correct; reset sends seed; success state terminates with bonus in reward; timeout truncates; `gymnasium.utils.env_checker.check_env` passes.
- [ ] **Step 2-4: Red, implement, green.** **Step 5: Commit.**

### Task 7: Capture

**Files:**
- Create: `src/teardown_lab/capture.py`
- Test: `tests/test_capture.py`

**Interfaces:**
- Produces: `screenshot(display: str, out_path: Path) -> Path` (mss); `EpisodeRecorder(display: str, out_dir: Path)` with `.start(name) / .stop()` spawning/terminating an ffmpeg x11grab subprocess; `ffmpeg_cmd(display, size, fps, out) -> list[str]` (pure, tested).

- [ ] **Step 1: Failing test** on `ffmpeg_cmd` exact argv (`-f x11grab -framerate 30 -video_size 1920x1080 -i :1 -c:v libx264 -preset veryfast -pix_fmt yuv420p <out>`).
- [ ] **Step 2-4: Red, implement, green** (subprocess start/stop behind an injectable `popen` for a lifecycle test). **Step 5: Commit.**

### Task 8: Lua mod + real bridge transport (game-side)

**Files:**
- Create: `mod/` (layout per Task 1 record), real transport class in `src/teardown_lab/bridge.py`
- Test: `tests/test_bridge_real.py` (`@pytest.mark.game`)

**Interfaces:**
- Consumes: Task 1 decision record verbatim.
- Produces: `RealBridge(**cfg)` satisfying `Bridge`; a mod exposing the 9-block tower level, streaming `GameState` JSON at >= 10 Hz, honoring `{"cmd":"reset","seed":n}` deterministically.

- [ ] **Step 1:** Implement the mod per the decision record: level with 9 tagged dynamic blocks (3-wide, 3-high stack), fixed player spawn; per-tick serialization of player + tagged block transforms; reset handler calling `SetRandomSeed(seed)` + `Restart()`.
- [ ] **Step 2:** Symlink the mod into the Proton mods dir; enable it in-game (screenshot-drive the UI once; document the clicks in the decision record).
- [ ] **Step 3:** Implement `RealBridge` for the chosen transport.
- [ ] **Step 4:** `@pytest.mark.game` smoke test: with the level running, `read_state()` yields parseable `GameState` with 9 blocks at >= 10 Hz for 5 s; reset with two different seeds yields identical initial block poses per seed (determinism check, two trials each).
- [ ] **Step 5:** Run `uv run pytest -m game tests/test_bridge_real.py` with the game up — green. **Commit** (mod + RealBridge + updated decision record).

### Task 9: Scripted policy end-to-end + recorded trace

**Files:**
- Create: `src/teardown_lab/scripted.py`, `tests/fixtures/trace_topple.jsonl`, `tests/test_referee_trace.py`

**Interfaces:**
- Consumes: `TeardownTowerEnv` with `RealBridge` + `X11Backend`.
- Produces: `scripted_topple(obs) -> np.ndarray` (walk toward tower centroid, swing); a recorded real episode trace (one `GameState.to_json()` per line); trace-replay test asserting the referee detects success on the real trace.

- [ ] **Step 1:** `scripted.py`: pure function of obs — move toward the tower's relative centroid, swing when within reach (centroid distance < 1.5 in obs units).
- [ ] **Step 2:** Runner `uv run python -m teardown_lab.scripted` (guarded by `GAME_DISPLAY`): runs one episode live, logs each state to the trace file, prints success/time. Iterate until the scripted policy topples the tower.
- [ ] **Step 3:** Commit the successful trace as a test fixture; `test_referee_trace.py` replays it through `referee` and asserts success fires and reward sum is positive (no game needed — this is the regression net for referee changes).
- [ ] **Step 4: Commit.** M1 done.

### Task 10: SAC training + staged checkpoints

**Files:**
- Create: `src/teardown_lab/train_sac.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `TeardownTowerEnv` (any bridge).
- Produces: `train(env, total_steps, run_dir, seed) -> list[Path]` saving SB3 SAC checkpoints at 0/25/50/100% of `total_steps` named `stage_000.zip`...`stage_100.zip` + `progress.csv` (episode returns); CLI `uv run python -m teardown_lab.train_sac --steps N --run-dir runs/<name>`.

- [ ] **Step 1: Failing test**: `train(fake_env, total_steps=200, ...)` (FakeBridge env) produces exactly 4 checkpoint files at the right step counts (assert via a checkpoint-callback unit hook, not wall-clock).
- [ ] **Step 2-4: Red, implement (SB3 `SAC("MlpPolicy", env)` + `CheckpointCallback`-style custom callback), green.** **Step 5: Commit.**

### Task 11: Eval protocol

**Files:**
- Create: `src/teardown_lab/eval.py`
- Test: `tests/test_eval.py`

**Interfaces:**
- Consumes: any `policy(obs) -> np.ndarray`, `TeardownTowerEnv`, optional `EpisodeRecorder`.
- Produces: `evaluate(policy, env, seeds: list[int], recorder=None) -> EvalResult`; `EvalResult.to_json()` schema:

```json
{"n_episodes": 20, "success_rate": 0.65, "mean_time_to_success": 21.4,
 "seeds": [...], "episodes": [{"seed": 1, "success": true, "t": 18.2, "return": 12.3, "video": "ep_001.mp4"}, ...],
 "notes": {"timescale": 1.0, "observation": "privileged-state"}}
```

  Default seed list: `list(range(1, 21))`. CLI: `uv run python -m teardown_lab.eval --policy <ckpt|scripted|random> --run-dir <dir>`.

- [ ] **Step 1: Failing tests** with FakeBridge env + scripted/random policies: result counts, JSON schema keys, per-episode seed propagation into `env.reset(seed=...)`, recorder called once per episode when provided.
- [ ] **Step 2-4: Red, implement, green.** **Step 5: Commit.** M2 code-complete; real training/eval runs are operations tracked in tasks/todo.md.

## Self-Review

- **Spec coverage:** M0 = Tasks 1,2,4,8; M1 = Tasks 3,5,6,7,9; M2 = Tasks 10,11. VLA baselines + showcase deliberately out (next plan). Honesty fields live in `EvalResult.notes`.
- **Type consistency:** `GameState`/`Action`/`Bridge`/`EnvConfig` names checked across Tasks 3-11.
- **Placeholders:** Task 1/8 game-side steps are research-gated by design; all host-side tasks carry concrete interfaces and test intent. No TBDs.
