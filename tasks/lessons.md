<!-- ABOUTME: Lessons captured while building teardown-agent-lab (per self-improvement loop). -->
<!-- ABOUTME: Add a lesson after any correction or painful discovery; keep them actionable. -->

# lessons

- **2026-08-01, Steam automation:** the Steam client GUI (steamwebhelper/CEF) is unstable
  under automation on this workstation - blank pages, mid-dialog crashes that silently eat
  state (a EULA acceptance was lost this way). Use the Steam console
  (`steam://open/console` -> `app_install <id>`, `steam://rungameid/<id>`) and poll
  `appmanifest_<id>.acf` (`StateFlags 4` = installed) instead of clicking dialogs.
  Proton game windows have WM_CLASS `steam_app_<appid>`, not the game's name.
- **2026-08-01, input injection:** Teardown ignores XTest-synthesized input entirely
  (pyautogui/xdotool): the pointer moves and UI shows hover, but clicks/keys never
  register - a silent no-op that looks like "the click didn't land". Kernel-level uinput
  (`python-evdev` `UInput` on `/dev/uinput`, writable via ACL, no sudo) DOES register.
  Any future "the game ignores my input" symptom: check the injection layer first, and
  give the game ~1.5 s to enumerate a freshly created uinput device.
- **2026-08-01, log.txt is a post-mortem channel, not a live one.** It stays 0 bytes
  during play and is flushed on shutdown - which is why `DebugPrint` looked like it
  reached nothing. After the game exits it contains `StateSwitch` lines, `Active mod:
  <name> (local-<folder>)`, and Lua errors. Read it AFTER quitting to diagnose a mod
  that misbehaved; never poll it during a run.
- **2026-08-01, clicking blind is expensive.** A blind `click:1743,90` meant for a
  character-select button landed on the main menu's Quit and killed a live session.
  Screenshot and confirm the expected screen before every click in a multi-step UI
  sequence; batching unverified clicks trades a few seconds for whole-session restarts.
- **2026-08-01, verify API symbols against the shipped defs, not the web docs.** The
  public docs list 608 functions; `<game>/data/script_defs.lua` declares 751 and is
  authoritative (it also proved `Spawn()` exists despite the modding page saying runtime
  spawning is impossible). `IsBodyHandle` was invented from a plausible guess and does
  not exist - grep `script_defs.lua` for every symbol before writing Lua.
- **2026-08-03, stale files masquerade as live state.** `savegame.xml` survives reboots,
  so after a restart it still held the previous session's payload - phase=live, ep=92 -
  and the readiness check happily declared the game in a level while it sat at the main
  menu. Any "is it running?" check over a FILE must require the tick counter to advance,
  never just that a well-formed value exists.
- **2026-08-03, match the window, not the class.** The game creates several X windows
  sharing WM_CLASS, including a 1x1 unmapped helper. Matching the first one made the
  visibility check test a window that can never be read, so the harness reported the game
  hidden while it ran in plain sight. Pick the largest MAPPED window.
- **2026-08-03, the X display number is not stable across reboots.** Greeter login gives
  :1, GDM autologin gives :0. Everything hardcoded :1 and silently addressed a dead
  display after a reboot. Detect it at runtime from /tmp/.X11-unix.
- **2026-08-03, the game itself is the fragile component.** Teardown hard-crashed
  mid-`Spawn` after ~30 h of session and ~90 episodes, with no error in log.txt - just a
  truncated line. Our reset spawns and deletes 9 bodies per episode, so a long run churns
  thousands of entities. Anything running unattended for an hour must supervise the GAME
  PROCESS (detect death, relaunch, drive back into a level, resume from checkpoint), not
  just the level. Steam also frequently refuses to relaunch the game afterwards and needs
  a full client restart.
- **2026-08-03, checkpoint long data collection.** 41 episodes were lost to that crash
  because the dataset was only written at the end. Any run measured in tens of minutes
  must persist incrementally.
- **2026-08-01, process matching:** `pgrep -f teardown` matched this session's own tmux
  session name and a kernel thread (`oom_reaper`); a "game process is up" signal was a
  false positive for ~5 min. Match full binary paths (`teardown.exe`) and confirm with a
  mapped X window, not just a process.
