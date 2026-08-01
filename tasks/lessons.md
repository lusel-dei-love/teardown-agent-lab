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
- **2026-08-01, process matching:** `pgrep -f teardown` matched this session's own tmux
  session name and a kernel thread (`oom_reaper`); a "game process is up" signal was a
  false positive for ~5 min. Match full binary paths (`teardown.exe`) and confirm with a
  mapped X window, not just a process.
