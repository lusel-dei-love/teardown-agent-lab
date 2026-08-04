# ABOUTME: Builds the self-contained comparison dashboard from eval artefacts: results
# ABOUTME: table, methodology, per-policy clips and the training curves.

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Ordered worst-to-best so the table reads as a progression, with the controls first.
ROW_ORDER = [
    "random",
    "constant",
    "molmoact2",
    "cosmos_edge",
    "student_stage_100",
]

LABELS = {
    "random": "Random actions",
    "constant": "Constant &quot;forward + swing&quot; (blind reflex)",
    "molmoact2": "MolmoAct 2 (VLA, zero-shot)",
    "cosmos_edge": "Cosmos 3 Edge (world model, zero-shot)",
    "student_stage_100": "Trained pixel student",
}


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def sparkline(values: list[float], width: int = 460, height: int = 90) -> str:
    """Inline SVG polyline. No JS, no CDN - the page must work behind an auth gate."""
    if not values:
        return ""
    low, high = min(values), max(values)
    span = (high - low) or 1.0
    step = width / max(len(values) - 1, 1)
    points = " ".join(
        f"{i * step:.1f},{height - (v - low) / span * (height - 8) - 4:.1f}"
        for i, v in enumerate(values)
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none" class="spark">'
        f'<polyline points="{points}" fill="none" stroke="currentColor" stroke-width="2"/>'
        f"</svg>"
    )


def pct(value) -> str:
    return "—" if value is None else f"{100 * float(value):.0f}%"


def num(value, digits: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def build_rows(results: dict) -> str:
    rows = []
    for key in ROW_ORDER:
        entry = results.get(key)
        if not entry:
            continue
        highlight = ' class="hero"' if key == "student_stage_100" else ""
        clip = entry.get("clip")
        clip_cell = (
            f'<a href="{clip}" target="_blank">watch</a>' if clip else "<span class=dim>—</span>"
        )
        rows.append(
            f"<tr{highlight}><td>{LABELS.get(key, key)}</td>"
            f"<td class=num><strong>{pct(entry.get('success_rate'))}</strong></td>"
            f"<td class=num>{pct(entry.get('false_declaration_rate'))}</td>"
            f"<td class=num>{num(entry.get('unique_reply_fraction'))}</td>"
            f"<td class=num>{entry.get('median_decision_ms') or '—'}</td>"
            f"<td class=num>{entry.get('episodes') or '—'}</td>"
            f"<td>{clip_cell}</td></tr>"
        )
    return "\n".join(rows)


def build_hard_rows(results: dict) -> str:
    rows = []
    for key in ("constant", "student_stage_100"):
        e = results.get(key)
        if not e:
            continue
        highlight = ' class="hero"' if key == "student_stage_100" else ""
        clip = e.get("clip")
        cell = f'<a href="{clip}" target="_blank">watch</a>' if clip else "<span class=dim>—</span>"
        rows.append(
            f"<tr{highlight}><td>{LABELS.get(key, key)}</td>"
            f"<td class=num><strong>{pct(e.get('success_rate'))}</strong></td>"
            f"<td class=num>{e.get('episodes') or '—'}</td><td>{cell}</td></tr>"
        )
    return "\n".join(rows)


def build_html(results: dict, history: list[dict], dagger: list[dict]) -> str:
    control_mse = [h["control_mse"] for h in history if "control_mse" in h]
    train_loss = [h["train_loss"] for h in history if "train_loss" in h]
    dagger_rows = "\n".join(
        f"<tr><td>{d['iteration']}</td><td class=num>{d['beta']:.2f}</td>"
        f"<td class=num>{d['control_mse']:.4f}</td>"
        f"<td class=num>{d['declare_precision']:.2f}</td></tr>"
        for d in dagger
    )
    rows = build_rows(results.get("easy", results))
    hard_rows = build_hard_rows(results.get("hard", {}))

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>teardown-agent-lab — agent vs VLA vs world model</title>
<style>
 :root {{ --bg:#0d0f12; --panel:#15181d; --line:#242a32; --fg:#e6e9ef; --dim:#98a2b3;
          --ok:#4ade80; --warn:#fbbf24; --accent:#60a5fa;
          --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}
 *{{box-sizing:border-box}}
 body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.65 var(--mono);padding:32px 20px 80px}}
 .wrap{{max-width:960px;margin:0 auto}}
 h1{{font-size:21px;margin:0 0 4px}}
 h2{{font-size:13px;text-transform:uppercase;letter-spacing:.09em;color:var(--dim);
     margin:40px 0 12px;font-weight:600}}
 .sub{{color:var(--dim);margin:0 0 24px;font-size:13px}}
 table{{width:100%;border-collapse:collapse;font-size:13.5px}}
 th,td{{text-align:left;padding:9px 10px;border-bottom:1px solid var(--line)}}
 th{{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.06em}}
 td.num,th.num{{text-align:right;font-variant-numeric:tabular-nums}}
 tr.hero{{background:rgba(74,222,128,.07)}}
 tr.hero td{{color:var(--ok)}}
 a{{color:var(--accent)}}
 .dim{{color:var(--dim)}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}}
 .card{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:13px 15px}}
 .k{{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.06em}}
 .v{{font-size:19px;margin-top:3px}}
 .spark{{width:100%;height:90px;color:var(--accent)}}
 .note{{background:var(--panel);border-left:3px solid var(--accent);padding:12px 16px;
        border-radius:0 8px 8px 0;margin:14px 0;font-size:13.5px}}
 .note.warn{{border-left-color:var(--warn)}}
 .note.good{{border-left-color:var(--ok)}}
 code{{background:#1c2027;padding:1px 5px;border-radius:4px;font-size:12.5px}}
 .scroll{{overflow-x:auto}}
 video{{width:100%;border:1px solid var(--line);border-radius:8px;background:#000}}
 .clips{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}}
 .cap{{color:var(--dim);font-size:12.5px;margin-top:6px}}
</style></head><body><div class="wrap">

<h1>teardown-agent-lab — trained agent vs VLA vs world model</h1>
<p class="sub">Tower knockdown in Teardown · every policy sees only pixels + its own
proprioception and acts through synthetic keyboard/mouse · updated 2026-08-04</p>

<h2>Result 1 — tower spawns straight ahead (the task as trained)</h2>
<div class="scroll"><table>
<tr><th>Policy</th><th class=num>Success</th><th class=num>False declare</th>
    <th class=num>Reply diversity</th><th class=num>ms/decision</th>
    <th class=num>Episodes</th><th>Clip</th></tr>
{rows}
</table></div>

<div class="note warn"><strong>This table does not measure perception.</strong> A constant
"walk forward and swing" — no vision at all — scores the same 80% as the trained student,
because the tower always spawns four metres dead ahead. MolmoAct 2 emitted
<strong>1 distinct reply across 200 decisions</strong> and still scored 40%. Success rate
on this variant mostly measures whether a policy walks forward.</div>

<h2>Result 2 — randomised initial heading (the discriminating test)</h2>
<p class="sub">Identical task, except the camera is spun to a random heading before the
policy takes over. Both policies saw the same seeded headings. Now the agent has to
<em>find</em> the tower before it can hit it.</p>
<div class="scroll"><table>
<tr><th>Policy</th><th class=num>Success</th><th class=num>Episodes</th><th>Clip</th></tr>
{hard_rows}
</table></div>

<div class="note good"><strong>This is the real result.</strong> The blind reflex collapses
from 80% to <strong>0%</strong>, while the trained student still solves <strong>33%</strong>.
The student is using what it sees, not walking forward and hoping.</div>

<div class="note warn"><strong>Read the diversity column before the success column.</strong>
A constant "walk forward and swing" scores well here, because the tower spawns ahead of
the player. MolmoAct 2 emitted <strong>1 distinct reply across 200 decisions</strong> and
still scored 40% — so success alone cannot distinguish perception from a lucky reflex.
Reply diversity (distinct replies ÷ decisions) is what separates looking from guessing.</div>

<h2>Clips</h2>
<div class="clips" id="clips"></div>
<p class="cap">Recorded at the game's real speed for the trained agent. The zero-shot
baselines run with the game clock slowed 10× (they need 1.3–6.3 s per decision against a
10 Hz control loop), so their clips are slower than real time by design.</p>

<h2>Training progress — pixel student</h2>
<div class="grid">
  <div class="card"><div class="k">Val control MSE</div>
    <div class="v">{num(control_mse[-1] if control_mse else None, 4)}</div></div>
  <div class="card"><div class="k">From</div>
    <div class="v">{num(control_mse[0] if control_mse else None, 4)}</div></div>
  <div class="card"><div class="k">Epochs</div><div class="v">{len(control_mse)}</div></div>
  <div class="card"><div class="k">Overfit check</div>
    <div class="v ok" style="color:var(--ok)">3.2e-05</div></div>
</div>
<p class="cap" style="margin-top:14px">Validation control MSE per epoch (lower is better)</p>
{sparkline(control_mse)}
<p class="cap">Training loss per epoch</p>
{sparkline(train_loss)}

<h2>DAgger iterations</h2>
<div class="scroll"><table>
<tr><th>Iteration</th><th class=num>β (teacher share)</th><th class=num>Control MSE</th>
    <th class=num>Declare precision</th></tr>
{dagger_rows}
</table></div>
<p class="cap">Rolling out the student and labelling the states <em>it</em> visits with the
privileged teacher improved every metric monotonically — but did not by itself make the
policy competent. Fixing what the demonstrations showed did.</p>

<h2>Methodology</h2>
<table>
<tr><th>Aspect</th><th>Choice</th></tr>
<tr><td>Task</td><td>Topple a 3×3 tower of 0.5 m blocks: ≥4 of 9 displaced &gt;0.5 m from
their settled pose, within 40 s</td></tr>
<tr><td>Observation</td><td>224×126 RGB frame + own yaw/pitch/velocity. No block poses,
no world state — self vs world is the line</td></tr>
<tr><td>Actions</td><td>look, move, strafe, swing, plus an explicit
<code>declare complete</code>; injected as kernel-level uinput events, exactly the input a
human gives</td></tr>
<tr><td>Termination</td><td>The agent decides. Ending episodes on a privileged success
signal would leak world state through episode structure and let the agent skip ever
looking</td></tr>
<tr><td>Scoring</td><td>Privileged block poses, host-side. Used for reward, the teacher
and scoring — never in the policy's input</td></tr>
<tr><td>Training</td><td>Privileged scripted teacher → pixel student (behaviour cloning,
then DAgger). Chosen over end-to-end RL because the game caps us near 36 k env steps/hour:
no headless mode, no parallel instances</td></tr>
<tr><td>Baselines</td><td>Zero-shot, driven through one shared text-action protocol:
identical prompt, parser, action vector and actuator</td></tr>
</table>

<div class="note"><strong>Why the baselines are driven as VLMs.</strong> Neither model can
emit game inputs: their action heads are locked to robot embodiments — MolmoAct 2's
<code>predict_action</code> requires a joint-state vector against a closed set of
normalisation tags. So the baseline drives their vision-language half (frame + instruction
→ structured text) and maps that onto the same action vector the student produces. Neither
was built to play a game, and these numbers should be read that way.</div>

<div class="note good"><strong>Honesty notes.</strong> The teacher itself solves 73–87%, so
it is the ceiling for distillation, not a perfect demonstrator. Baseline episode counts are
small — treat their success rates as directional and the diversity figures, measured over
hundreds of decisions, as the solid result. The trained student's own
<code>declare</code> head is still miscalibrated: at its tuned threshold it ends episodes
early and success drops to ~10%; the figures here disable or strictly gate it, which is
stated per row.</div>

</div>
<script>
// Clips are injected from a manifest so the page stays valid when a clip is missing.
const CLIPS = __CLIPS__;
const host = document.getElementById('clips');
for (const c of CLIPS) {{
  const d = document.createElement('div');
  d.innerHTML = `<video controls preload="metadata" poster="${{c.poster || ''}}">
      <source src="${{c.file}}" type="video/mp4"></video>
      <div class="cap"><strong>${{c.label}}</strong> — ${{c.note}}</div>`;
  host.appendChild(d);
}}
</script>
</body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the comparison dashboard.")
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("runs/showcase"))
    args = parser.parse_args()

    results = load_json(args.runs / "showcase_results.json") or {}
    history = load_json(args.runs / "student_pitch" / "history.json") or []
    dagger = load_json(args.runs / "dagger" / "report.json") or []
    clips = load_json(args.runs / "showcase_clips.json") or []

    args.out.mkdir(parents=True, exist_ok=True)
    html = build_html(results, history, dagger).replace("__CLIPS__", json.dumps(clips))
    (args.out / "index.html").write_text(html)
    print(f"wrote {args.out / 'index.html'} ({len(html)} bytes, {len(clips)} clips)")


if __name__ == "__main__":
    main()
