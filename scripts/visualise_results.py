#!/usr/bin/env python3
import argparse, re, os, json, glob
from pathlib import Path
from html import escape
from datetime import datetime

TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8"><title>Animal Sound Results</title>
<style>
  :root{{--bg:#0b1020;--card:#111830;--muted:#9fb3c8;--text:#e9eef7;--accent:#5ac8fa;}}
  *{{box-sizing:border-box}}
  body{{margin:0; padding:26px; font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif; color:var(--text); background:linear-gradient(180deg,#0b1020,#070b16)}}
  h1{{font-size:26px; margin:0 0 16px}}
  h2{{font-size:18px; margin:20px 0 12px}}
  .muted{{color:var(--muted)}}
  .grid{{display:grid; gap:16px}}
  .g-2{{grid-template-columns:repeat(2,minmax(0,1fr))}}
  .g-3{{grid-template-columns:repeat(3,minmax(0,1fr))}}
  @media (max-width:1000px){{.g-2,.g-3{{grid-template-columns:1fr}}}}
  .card{{background:var(--card); border:1px solid rgba(255,255,255,.06); border-radius:14px; padding:16px; box-shadow:0 6px 18px rgba(0,0,0,.25)}}
  .row{{display:flex; gap:16px; align-items:flex-start; flex-wrap:wrap}}
  img{{max-width:100%; border-radius:10px; border:1px solid rgba(255,255,255,.08)}}
  table{{border-collapse:collapse; width:100%; overflow:auto; display:block}}
  th,td{{padding:8px 10px; border-bottom:1px solid rgba(255,255,255,.08); text-align:right}}
  th:first-child,td:first-child{{text-align:left}}
  thead th{{position:sticky; top:0; background:rgba(255,255,255,.05)}}
  .kpi{{display:grid; grid-template-columns:repeat(3,1fr); gap:12px}}
  .pill{{background:rgba(90,200,250,.12); color:#aee6ff; border:1px solid rgba(90,200,250,.35); padding:4px 8px; border-radius:999px}}
  .footer{{margin-top:18px; color:var(--muted); font-size:12px}}
</style>
</head>
<body>

<h1>Animal Sound Decoding — Results</h1>
<div class="muted">Generated: {ts}</div>

<div class="grid g-2">
  <section class="card">
    <h2>Evaluation Summary</h2>
    {report_table}
    <div class="row" style="margin-top:10px">
      {kpis}
    </div>
  </section>

  <section class="card">
    <h2>Confusion Matrix</h2>
    {cm_img}
  </section>
</div>

<section class="card" style="margin-top:16px">
  <h2>Training Curve</h2>
  {loss_img}
</section>

<section class="card" style="margin-top:16px">
  <h2>Waveforms (concatenated, colored)</h2>
  <div class="grid g-3">
    {wave_imgs}
  </div>
</section>

<section class="card" style="margin-top:16px">
  <h2>MFCC Heatmaps</h2>
  <div class="grid g-3">
    {mfcc_grids}
  </div>
</section>

<div class="footer">This report is static HTML (no JS) so it renders anywhere. Paths are relative to the project root.</div>
</body></html>
"""

def parse_classification_report(txt: str):
    """Parse sklearn classification_report text into headers+rows and KPI dict."""
    lines = [l.rstrip() for l in txt.splitlines() if l.strip()]
    start = None
    for i, l in enumerate(lines):
        if re.search(r'^\s*precision\s+recall\s+f1-score\s+support', l):
            start = i; break
    if start is None:
        return None, None, {}
    headers = re.split(r'\s{2,}', lines[start].strip())
    rows, kpis = [], {}
    for l in lines[start+1:]:
        s = l.strip()
        if s.startswith('accuracy'):
            m = re.findall(r'([0-9.]+)', s)
            if m: kpis['accuracy'] = float(m[0])
            continue
        if s.startswith('macro avg'):
            vals = re.split(r'\s{2,}', s)
            kpis['macro_precision']=float(vals[1]); kpis['macro_recall']=float(vals[2]); kpis['macro_f1']=float(vals[3]); continue
        if s.startswith('weighted avg'):
            vals = re.split(r'\s{2,}', s)
            kpis['weighted_precision']=float(vals[1]); kpis['weighted_recall']=float(vals[2]); kpis['weighted_f1']=float(vals[3]); continue
        parts = re.split(r'\s{2,}', s)
        if len(parts) >= 5:
            cls, prec, rec, f1, sup = parts[0:5]
            rows.append([cls, prec, rec, f1, sup])
    return headers, rows, kpis

def table_html(headers, rows):
    if not headers or not rows:
        return '<pre class="muted">No classification table parsed.</pre>'
    th = ''.join(f'<th>{escape(h)}</th>' for h in ['class'] + headers[1:])
    trs = []
    for r in rows:
        tds = ''.join(f'<td>{escape(c)}</td>' for c in r)
        trs.append(f'<tr>{tds}</tr>')
    return f'<table><thead><tr>{th}</tr></thead><tbody>{"".join(trs)}</tbody></table>'

def img_tag(path, alt):
    if not Path(path).exists():
        return f'<div class="muted">Missing: {escape(path)}</div>'
    return f'<img src="{escape(path)}" alt="{escape(alt)}" />'

def species_wave_imgs(project_root, species=("cat","lion","dog","bird","monkey")):
    out = []
    for sp in species:
        p = Path(project_root) / f"results/{sp}_waveforms_concat_colored.png"
        out.append(f'<div><div class="pill">{escape(sp)}</div>{img_tag(str(p), f"{sp} waveform")}</div>')
    return "\n".join(out)

def mfcc_grids_html(project_root, species=("cat","lion","dog")):
    blocks = []
    for sp in species:
        folder = Path(project_root) / f"results/mfcc_plots/{sp}"
        imgs = sorted(glob.glob(str(folder / "mfcc_*.png")))
        if not imgs:
            blocks.append(f'<div><div class="pill">{escape(sp)}</div><div class="muted">No MFCC images.</div></div>')
            continue
        cards = ''.join(f'<div>{img_tag(i, Path(i).name)}</div>' for i in imgs)
        blocks.append(f'<div><div class="pill">{escape(sp)}</div><div class="grid g-3" style="margin-top:8px">{cards}</div></div>')
    return "\n".join(blocks)

def kpis_html(kpis: dict):
    if not kpis: return ''
    def pct(x): return f"{x*100:.1f}%"
    parts = []
    if 'accuracy' in kpis: parts.append(f'<div class="pill">Accuracy: {pct(kpis["accuracy"])}</div>')
    if 'macro_f1' in kpis: parts.append(f'<div class="pill">Macro F1: {pct(kpis["macro_f1"])}</div>')
    if 'weighted_f1' in kpis: parts.append(f'<div class="pill">Weighted F1: {pct(kpis["weighted_f1"])}</div>')
    return ''.join(parts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report_txt", required=True)
    ap.add_argument("--cm_png", required=True)
    ap.add_argument("--out_html", required=True)
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--species", nargs="+", default=["cat","lion","dog","bird","monkey"])
    args = ap.parse_args()

    report_txt = Path(args.report_txt).read_text(encoding="utf-8", errors="ignore")
    headers, rows, kpis = parse_classification_report(report_txt)

    html = TEMPLATE.format(
        ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        report_table=table_html(headers, rows),
        kpis=kpis_html(kpis),
        cm_img=img_tag(args.cm_png, "Confusion Matrix"),
        loss_img=img_tag("results/loss_curve.png", "Training / Validation Loss"),
        wave_imgs=species_wave_imgs(args.project_root, tuple(args.species)),
        mfcc_grids=mfcc_grids_html(args.project_root, tuple(args.species)),
    )

    Path(args.out_html).write_text(html, encoding="utf-8")
    print(f"[OK] wrote {args.out_html}")

if __name__ == "__main__":
    main()
