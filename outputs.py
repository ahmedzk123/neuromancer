"""outputs.py -- prediction.json, ostia.csv, and check.html writers."""
from __future__ import annotations

import csv
import json

import numpy as np


def build_prediction_dict(case_id, candidates):
    daughters = []
    for c in candidates:
        daughters.append({
            "instance_id": c["instance_id"],
            "parent_instance_id": "aorta",
            "ostium_xyz_mm": [round(float(v), 3) for v in c["ostium_mm"]],
            "seed_xyz_mm": [round(float(v), 3) for v in c["seed_mm"]],
            "radius_mm": round(float(c["radius_mm"]), 3),
            "direction_xyz": [round(float(v), 4) for v in c["direction_xyz"]],
        })
    return {"case_id": case_id, "parent": {"instance_id": "aorta"},
            "daughters": daughters}


def write_prediction_json(path, case_id, candidates):
    with open(path, "w") as fh:
        json.dump(build_prediction_dict(case_id, candidates), fh, indent=2)


def write_ostia_csv(path, candidates):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["instance_id", "ostium_x_mm", "ostium_y_mm", "ostium_z_mm",
                    "seed_x_mm", "seed_y_mm", "seed_z_mm",
                    "dir_x", "dir_y", "dir_z", "radius_mm"])
        for c in candidates:
            w.writerow([c["instance_id"], *[round(float(v), 3) for v in c["ostium_mm"]],
                        *[round(float(v), 3) for v in c["seed_mm"]],
                        *[round(float(v), 4) for v in c["direction_xyz"]],
                        round(float(c["radius_mm"]), 3)])


def _hu_along_trace(ctx, path_vox):
    """Sample CT intensity along the traced path, for the clinician QA inset."""
    vals, dist = [], []
    cum = 0.0
    prev = None
    for p in path_vox:
        zz, yy, xx = [int(round(v)) for v in p]
        zz = max(0, min(zz, ctx.ct.shape[0] - 1))
        yy = max(0, min(yy, ctx.ct.shape[1] - 1))
        xx = max(0, min(xx, ctx.ct.shape[2] - 1))
        vals.append(float(ctx.ct[zz, yy, xx]))
        if prev is not None:
            cum += float(np.linalg.norm((np.array(p) - np.array(prev)) * ctx.sp))
        dist.append(round(cum, 2))
        prev = p
    return dist, vals


def write_check_html(path, case_id, ctx, mk_full_shape, candidates, grown_mask):
    """Self-contained interactive check: aorta + ostia + direction arrows in 3D,
    plus a per-branch HU-along-trace inset (the clinician QA idea) so a reviewer
    can see at a glance whether a detected 'vessel' has a plausible contrast-
    filled lumen the whole way, not just a point in the right place.
    """
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        with open(path, "w") as fh:
            fh.write(f"<title>{case_id}</title><p>scikit-image not installed -- "
                     f"no 3D check available.</p>")
        return

    def surface(binary, target_vox=1.0):
        if not binary.any():
            return None, None
        from scipy import ndimage as ndi
        sx, sy, sz = ctx.sx, ctx.sy, ctx.sz
        zoom = (sz / target_vox, sy / target_vox, sx / target_vox)
        small = ndi.zoom(binary.astype(np.float32), zoom, order=1)
        small = ndi.gaussian_filter(small, 0.7)
        if small.max() < 0.5:
            return None, None
        v, f, _, _ = marching_cubes(small, level=0.5, spacing=(target_vox,) * 3)
        v = v[:, [2, 1, 0]]                            # (z,y,x)mm -> (x,y,z)mm
        # axis-aligned placement relative to the crop corner -- an approximation
        # for oblique (non-identity direction-cosine) volumes, fine for this
        # verification view since it is not used for scoring
        corner = ctx.to_mm_continuous((0, 0, 0))
        return corner + v, f

    av, af = surface(ctx.mk)
    gv, gf = surface(grown_mask)

    traces = []
    for c in candidates:
        dist, vals = _hu_along_trace(ctx, c["path_vox"])
        traces.append({
            "id": c["instance_id"],
            "ostium": [round(float(v), 2) for v in c["ostium_mm"]],
            "seed": [round(float(v), 2) for v in c["seed_mm"]],
            "direction": [round(float(v), 3) for v in c["direction_xyz"]],
            "radius": round(float(c["radius_mm"]), 2),
            "hu_dist": dist, "hu_val": vals,
        })

    data = {
        "case_id": case_id,
        "aorta_verts": av.tolist() if av is not None else [],
        "aorta_faces": af.tolist() if af is not None else [],
        "grown_verts": gv.tolist() if gv is not None else [],
        "grown_faces": gf.tolist() if gf is not None else [],
        "branches": traces,
    }

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{case_id} check</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js"></script>
<style>
 body{{font-family:system-ui,sans-serif;margin:0;display:flex;height:100vh}}
 #scene{{flex:3}} #panel{{flex:1;overflow-y:auto;padding:12px;border-left:1px solid #ddd}}
 h3{{margin:4px 0}} .branch{{border:1px solid #ddd;border-radius:6px;padding:8px;margin-bottom:10px;cursor:pointer}}
 .branch:hover{{background:#f5f5f5}} .hu{{width:100%;height:90px}}
</style></head><body>
<div id="scene"></div>
<div id="panel"><h2>{case_id}</h2><p>loading...</p><div id="branches"></div></div>
<script>
const DATA = {json.dumps(data)};
const traces = [];
if (DATA.aorta_verts.length) {{
  const v = DATA.aorta_verts, f = DATA.aorta_faces;
  traces.push({{type:'mesh3d', x:v.map(p=>p[0]), y:v.map(p=>p[1]), z:v.map(p=>p[2]),
    i:f.map(t=>t[0]), j:f.map(t=>t[1]), k:f.map(t=>t[2]),
    color:'#d94a4a', opacity:0.35, name:'aorta', hoverinfo:'skip'}});
}}
if (DATA.grown_verts.length) {{
  const v = DATA.grown_verts, f = DATA.grown_faces;
  traces.push({{type:'mesh3d', x:v.map(p=>p[0]), y:v.map(p=>p[1]), z:v.map(p=>p[2]),
    i:f.map(t=>t[0]), j:f.map(t=>t[1]), k:f.map(t=>t[2]),
    color:'#2f9e9e', opacity:0.55, name:'grown candidates', hoverinfo:'skip'}});
}}
for (const b of DATA.branches) {{
  traces.push({{type:'scatter3d', mode:'markers', x:[b.ostium[0]], y:[b.ostium[1]], z:[b.ostium[2]],
    marker:{{size:6, color:'#f2c14e'}}, name:b.id + ' ostium', text:[b.id], hoverinfo:'text'}});
  const tip = b.ostium.map((v,i)=>v + b.direction[i]*8);
  traces.push({{type:'scatter3d', mode:'lines', x:[b.ostium[0],tip[0]], y:[b.ostium[1],tip[1]], z:[b.ostium[2],tip[2]],
    line:{{color:'#111', width:5}}, name:b.id + ' direction', hoverinfo:'skip'}});
}}
Plotly.newPlot('scene', traces, {{scene:{{aspectmode:'data'}}, margin:{{l:0,r:0,t:30,b:0}},
  title:DATA.case_id + ' -- aorta + detected daughters'}});

const panel = document.getElementById('branches');
document.querySelector('#panel p').textContent = DATA.branches.length + ' daughter(s) detected';
for (const b of DATA.branches) {{
  const div = document.createElement('div');
  div.className = 'branch';
  div.innerHTML = `<h3>${{b.id}}</h3>
    <div>ostium: ${{b.ostium.join(', ')}} mm</div>
    <div>seed: ${{b.seed.join(', ')}} mm  |  radius: ${{b.radius}} mm</div>
    <canvas class="hu" id="hu_${{b.id}}"></canvas>`;
  panel.appendChild(div);
}}
// tiny inline HU-vs-distance plot per branch, no extra chart library needed
for (const b of DATA.branches) {{
  const c = document.getElementById('hu_' + b.id);
  const ctx2 = c.getContext('2d');
  c.width = c.clientWidth || 260; c.height = 90;
  const xs = b.hu_dist, ys = b.hu_val;
  if (!xs.length) continue;
  const xmin=Math.min(...xs), xmax=Math.max(...xs,0.001);
  const ymin=Math.min(...ys,0), ymax=Math.max(...ys,1);
  ctx2.strokeStyle = '#2f9e9e'; ctx2.beginPath();
  xs.forEach((x,i)=>{{
    const px = (x-xmin)/(xmax-xmin+1e-9) * (c.width-10) + 5;
    const py = c.height - 5 - (ys[i]-ymin)/(ymax-ymin+1e-9) * (c.height-10);
    i===0 ? ctx2.moveTo(px,py) : ctx2.lineTo(px,py);
  }});
  ctx2.stroke();
}}
</script></body></html>"""
    with open(path, "w") as fh:
        fh.write(html)
