<!-- ABOUTME: Decision record for the Teardown <-> host bridge: measured transport choice, -->
<!-- ABOUTME: rejected alternatives, exact paths and Lua calls. Settles plan Task 1. -->

# Bridge transport decision (Teardown 2.0.4, Linux/Proton)

**Decided 2026-08-01. Measured on the live game, not inferred from docs.**

## Decision

**State out (game -> host): the registry, polled from `savegame.xml`.**
The mod writes with `SetString`/`SetInt`/`SetFloat`/`SetBool` under `savegame.mod.*`;
the game flushes the registry to `savegame.xml` continuously, and the host polls that
file. **Measured >= 49 Hz** of distinct value updates (that was the *polling* ceiling at
a 20 ms sample interval, so the true rate is at least that — it tracks frame rate).
The task needs 10 Hz, so this has ~5x headroom.

**Commands in (host -> game): synthetic key events via uinput**, read mod-side with
`InputPressed`. No file or registry channel is needed for the small command vocabulary
(reset, seed selection); the actuator already injects input for the policy anyway.

## Exact paths and shapes

- Registry file (poll this):
  `<prefix>/drive_c/users/steamuser/AppData/Local/Teardown/savegame.xml`
  where `<prefix>` = `~/.steam/steam/steamapps/compatdata/1167630/pfx`
- Mods dir (install the mod here; a symlink from the repo's `mod/` works):
  `<prefix>/drive_c/users/steamuser/Documents/Teardown/mods/<modname>/`
  with `info.txt` + `main.lua` in the mod root.
- **The game prefixes local mods with `local-` in the registry tree.** A mod folder
  named `bridgeprobe` writing `SetInt("savegame.mod.probe.ticks", n)` appears as:

```xml
<registry><savegame><mod>
  <local-bridgeprobe><probe>
    <status value="init"/>
    <ticks value="3302"/>
    <payload value="3302|55.273"/>
    <sawcmd value="0"/>
  </probe></local-bridgeprobe>
</mod></savegame></registry>
```

  So the host must look under `savegame/mod/local-<modname>/...`, NOT `savegame/mod/...`.
- Serialization plan for the real bridge: one `SetString` holding the whole `GameState`
  as compact JSON (single key = single atomic-ish read; avoids stitching values that
  were flushed at different ticks). Include the tick counter in the payload so the host
  can detect staleness and drop duplicate reads.

## Rejected alternatives (measured, not assumed)

- **HTTP / sockets — do not exist.** The Lua API has no network functions at all.
- **File writing — does not exist.** The only file function is `HasFile(path)`
  (existence check). `HasFile("MOD/cmd.txt")` did evaluate correctly in the probe
  (returned false with no file present), so it is a viable *in*-channel if ever needed,
  at 1 bit per filename.
- **`DebugPrint` -> `log.txt` — dead channel.** `AppData/Local/Teardown/log.txt` exists
  but stayed **0 bytes** after several thousand `DebugPrint` calls across a full level
  session. Do not build on it. (Presumably needs a dev/console build.)
- **Encoding state in on-screen pixels** via `UiText`/`UiRect` + screen capture: not
  needed given the registry works; keep as a last-resort fallback only.

## Mod enable flow (UI, one-time per mod)

Main menu -> `Play` -> `Mod manager` -> the mod appears under **Local files** -> click its
toggle (it turns yellow). Sandbox levels are locked by default; enable
`Options -> Game -> Sandbox -> Unlock all levels` to reach them. Entering a sandbox level
goes through a character-selection screen before the level loads.

**All UI interaction must use uinput** — see the plan's Task 5 note: Teardown ignores
XTest-synthesized clicks and keys entirely.

## Consequences for the implementation

- `RealBridge` polls `savegame.xml`, parses the `local-<mod>` subtree, decodes the JSON
  payload, and de-duplicates on the tick counter. No sockets, no log tailing.
- Reset: host injects a hotkey; the mod's `InputPressed` handler calls
  `SetRandomSeed(seed)` + `Restart()`. The seed can be advanced mod-side per reset and
  echoed back in the state payload so the host records which seed actually ran.
- Polling a file at 10 Hz is cheap, but read the whole file and parse once per tick;
  do not re-read per key.
