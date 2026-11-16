"""Gallery page: a local static index of pressed releases.

Art direction: "the pressing floor": lacquer black, molten wax red,
hand-set grotesk type, grooved concentric rules like a record face.
Deliberately one-off; not a SaaS template.
"""
from __future__ import annotations

import json
from pathlib import Path

CSS = """
:root { --wax:#8f1d1d; --lacquer:#120608; --bone:#e8d8c0; --groove:#2a0f12; }
* { box-sizing:border-box; }
body { margin:0; background:var(--lacquer); color:var(--bone);
  font-family:"Courier New",monospace; }
header { padding:4rem 6vw 2rem; border-bottom:3px double var(--wax); }
h1 { font-size:clamp(2.2rem,6vw,4.5rem); letter-spacing:-.04em;
  text-transform:uppercase; margin:0; color:var(--wax);
  text-shadow:0 0 24px rgba(143,29,29,.5); }
.tagline { margin-top:.6rem; opacity:.7; font-size:.95rem; }
.platter { position:relative; padding:3rem 6vw; }
.platter::before { content:""; position:absolute; inset:0;
  background:repeating-radial-gradient(circle at 85% -10%,
    transparent 0 14px, var(--groove) 14px 15px); opacity:.5;
  pointer-events:none; }
.release { position:relative; border:1px solid var(--groove);
  border-left:6px solid var(--wax); padding:1.4rem 1.6rem;
  margin:0 0 1.2rem; background:rgba(255,255,255,.015); }
.release h2 { margin:0 0 .4rem; font-size:1.5rem; letter-spacing:.02em; }
.meta { font-size:.85rem; opacity:.75; }
.stems { margin-top:.7rem; display:flex; flex-wrap:wrap; gap:.4rem; }
.stem { border:1px dashed var(--wax); padding:.15rem .55rem;
  font-size:.75rem; text-transform:uppercase; letter-spacing:.1em; }
a { color:var(--bone); }
a:hover { color:var(--wax); }
footer { padding:2rem 6vw 3rem; font-size:.75rem; opacity:.5; }
"""


def render_gallery(pack_manifests: list[dict], out_path: Path) -> Path:
    cards = []
    for m in pack_manifests:
        stems = "".join(f'<span class="stem">{n}</span>'
                        for n in m.get("stems", {}))
        cards.append(f"""
  <div class="release">
    <h2>{m.get('title', m.get('slug', '?'))}</h2>
    <div class="meta">{m.get('artist', '')} &middot;
      {m.get('bpm') or '?'} BPM &middot; {m.get('key') or '?'} &middot;
      pressed {m.get('generated_at', '')[:10]}</div>
    <div class="stems">{stems}</div>
  </div>""")
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Pressing Plant gallery</title>
<style>{CSS}</style></head><body>
<header><h1>Pressing Plant</h1>
<div class="tagline">Fresh lacquers. Stems for remixers.</div>
</header>
<div class="platter">{''.join(cards) or '<p>No pressings yet.</p>'}</div>
<footer>Cut on the floor. Hand-set, hot wax.</footer>
</body></html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path


def collect_manifests(out_root: Path) -> list[dict]:
    found = []
    staging = out_root / "staging"
    if staging.is_dir():
        for mf in sorted(staging.glob("*/manifest.json")):
            found.append(json.loads(mf.read_text()))
    return found
