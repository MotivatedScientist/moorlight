"""Publication figures of the current view: framing, carving, labels as SVG, legend draft, PyMOL script.

The browser renders the image (the same scene, with figure materials and maps as nested transparent surfaces); this
module decides what to draw and writes the files:

    <name>.png           with labels and distances drawn in
    <name>_plain.png     without text
    <name>.svg           the plain PNG plus labels as editable text (Illustrator, Inkscape)
    <name>_uncarved.png  maps not cut to the shown atoms (when carved), to check that carving hides nothing
    <name>.json          everything needed to make the figure again
    <name>_legend.md     a draft legend: maps, levels, resolution, carving
    <name>.pml + maps    with pymol=true: the same view in PyMOL
"""
from __future__ import annotations

import base64
import html
import json
import os
import re

import numpy as np

FOV = 30.0                                   # viewer camera field of view (degrees)
PRESETS = {                                  # width in mm, aspect (None: as on screen)
    "single": (89.0, None), "one-and-half": (120.0, None), "double": (183.0, None), "slide": (254.0, 16 / 9),
}
LEVEL_STEPS = (1.0, 1.5, 2.25)               # nested surfaces at these multiples of the map's level
OPACITY = (0.22, 0.38, 0.6)                  # outer (weak) to inner (strong)
PALETTES = {
    "standard": {"pos": "#1fa01f", "neg": "#d42020"},
    "cvd": {"pos": "#0072b2", "neg": "#e69f00"},     # Okabe-Ito blue / orange, readable with red-green colour blindness
}


def slug(text, default="figure"):
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(text or "")).strip("_")
    return (s[:48] or default).lower()


def frame(view, aspect, screen_aspect):
    """Half width / height (Å) of the figure at the focus, showing what the smaller screen dimension shows."""
    half_min = float(view.get("zoom", 30)) * np.tan(np.radians(FOV / 2)) * min(1.0, float(screen_aspect or 1))
    return (half_min * aspect, half_min) if aspect >= 1 else (half_min, half_min / aspect)


def view_axes(orient):
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat(orient).as_matrix() if orient else np.eye(3)


def shown_atoms(session, center, axes, half_w, half_h, slab):
    """xyz of atoms of visible molecules inside the figure's box (for carving), with their molecule ids."""
    pts, ids = [], []
    for m in session.molecules.values():
        if not m.visible or not len(m.xyz):
            continue
        v = (m.xyz - center) @ axes.T
        keep = (np.abs(v[:, 0]) < half_w + 1.5) & (np.abs(v[:, 1]) < half_h + 1.5) & (np.abs(v[:, 2]) < slab)
        if keep.any():
            pts.append(m.xyz[keep])
            ids.append(m.id)
    return (np.concatenate(pts) if pts else np.zeros((0, 3))), ids


def hidden_by_carving(x, center, axes, half_w, half_h, slab, atoms, carve):
    """Strongest |value| (sigma) of map x inside the figure's box but farther than `carve` from every shown atom."""
    from scipy.spatial import cKDTree
    r = float(min(30.0, np.sqrt(half_w ** 2 + half_h ** 2 + slab ** 2)))
    arr, origin, step = x.sigma_box(center, r, max_n=80)
    n = arr.shape[0]
    g = np.indices(arr.shape).reshape(3, -1).T * step + origin
    v = (g - center) @ axes.T
    inside = (np.abs(v[:, 0]) < half_w) & (np.abs(v[:, 1]) < half_h) & (np.abs(v[:, 2]) < slab)
    if not inside.any() or not len(atoms):
        return None
    d, _ = cKDTree(atoms).query(g[inside], distance_upper_bound=carve)
    far = ~np.isfinite(d)
    if not far.any():
        return None
    vals = arr.reshape(-1)[inside][far]
    vals = vals if x.kind == "diff" else np.maximum(vals, 0)
    k = int(np.argmax(np.abs(vals)))
    p = g[inside][far][k]
    return float(vals[k]), [round(float(c), 2) for c in p]


def write_files(out_dir, name, result, spec):
    """Write PNGs, SVG and spec from the browser's render. Returns the paths."""
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, name)
    paths = {}
    for key, suffix in (("png", ".png"), ("plain", "_plain.png"), ("uncarved", "_uncarved.png")):
        data = result.get(key)
        if isinstance(data, str) and "," in data:
            with open(base + suffix, "wb") as fh:
                fh.write(base64.b64decode(data.split(",", 1)[1]))
            paths[key] = base + suffix
    W, H = result["width"], result["height"]
    mm_w, mm_h = spec["size_mm"]
    fs = result.get("font_px", 24)
    color = "#111111" if spec["background"] != "black" else "#f2f2f2"
    texts = []
    for lb in result.get("labels", []):
        cls = "distance" if lb.get("kind") == "distance" else "label"
        anchor = "middle" if cls == "distance" else "start"
        texts.append(f'    <text class="{cls}" x="{lb["x"]:.1f}" y="{lb["y"]:.1f}" text-anchor="{anchor}">'
                     f'{html.escape(lb["text"])}</text>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
           f'width="{mm_w:.2f}mm" height="{mm_h:.2f}mm" viewBox="0 0 {W} {H}">\n'
           f'  <image id="image" x="0" y="0" width="{W}" height="{H}" xlink:href="{os.path.basename(base)}_plain.png"/>\n'
           f'  <g id="labels" font-family="Helvetica, Arial, sans-serif" font-size="{fs:.1f}" fill="{color}" '
           f'dominant-baseline="middle">\n' + "\n".join(texts) + "\n  </g>\n</svg>\n")
    with open(base + ".svg", "w") as fh:
        fh.write(svg)
    paths["svg"] = base + ".svg"
    with open(base + ".json", "w") as fh:
        json.dump(spec, fh, indent=1)
    paths["spec"] = base + ".json"
    return paths


def legend(spec, maps_info, models_info):
    """Draft legend text: what is shown, at which levels, from which data, with carving stated."""
    lines = []
    title = spec.get("title") or ""
    text = (spec.get("text") or "").strip()
    head = f"**{title}.** " if title else ""
    lines.append(head + (text if text else "_(what the figure shows)_"))
    mods = "; ".join(f"{m['name']} (carbon {m['carbon']})" for m in models_info)
    if mods:
        lines.append(f"Models: {mods}.")
    for x in maps_info:
        lv = ", ".join(f"{v:g}" for v in x["levels"])
        sign = "±" if x["kind"] == "diff" else ""
        what = "difference map" if x["kind"] == "diff" else "electron density map"
        cols = "green (+) / red (−)" if spec["palette"] == "standard" else "blue (+) / orange (−)"
        lines.append(f"{x['name']} ({what}{', ' + str(x['resolution']) + ' Å' if x.get('resolution') else ''}, "
                     f"from {x['source']}) shown as nested surfaces at {sign}{lv} σ "
                     f"(σ = rms of the whole map, {x['rms']:.3g} map units)"
                     + (f", {cols}" if x["kind"] == "diff" else "") + ".")
    if spec.get("carve"):
        lines.append(f"Density is shown within {spec['carve']:g} Å of the displayed atoms"
                     + (" (an uncarved version is provided)." if spec.get("uncarved") else "."))
    if spec.get("distances"):
        lines.append("Distances in Å.")
    return "\n\n".join(lines) + "\n"


def pymol_script(session, out_dir, name, spec, maps, half_w, half_h):
    """A .pml that opens the same view in PyMOL: models (as loaded or saved), maps cut to the figure region (CCP4),
    isosurfaces at the same levels, carved like the figure."""
    import gemmi
    lines = ["# PyMOL script written by Moorlight: the same view and maps as " + name + ".png",
             "reinitialize", "set bg_rgb, " + ("white" if spec["background"] != "black" else "black"),
             "set orthoscopic, on", f"set field_of_view, {FOV:g}", "set normalize_ccp4_maps, 0",
             "set ray_shadows, off", "set transparency_mode, 3", "set surface_quality, 1", "set depth_cue, 1"]
    sel_parts = []
    for m in session.molecules.values():
        if not m.visible:
            continue
        path = os.path.join(out_dir, f"{name}_model{m.id}.pdb")
        m.write(path)
        obj = f"m{m.id}"
        lines += [f"load {os.path.basename(path)}, {obj}", f"hide everything, {obj}",
                  f"show sticks, {obj} and not hydro", f"show nb_spheres, {obj} and (resn HOH or solvent)",
                  f"show spheres, {obj} and metals", f"set sphere_scale, 0.35, {obj} and metals",
                  f"color {spec['carbon'].get(str(m.id), 'grey70').replace('#', '0x')}, {obj} and elem C"]
        sel_parts.append(obj)
    c = np.asarray(spec["center"])
    r = float(min(30.0, np.sqrt(half_w ** 2 + half_h ** 2 + spec["slab"] ** 2)) + 2)
    for x, levels in maps:
        fn = f"{name}_map{x.id}.ccp4"
        ccp4 = gemmi.Ccp4Map()
        ccp4.grid = x.grid
        ccp4.update_ccp4_header()
        box = gemmi.FractionalBox()
        for dx in (-r, r):
            for dy in (-r, r):
                for dz in (-r, r):
                    box.extend(x.grid.unit_cell.fractionalize(gemmi.Position(*(c + [dx, dy, dz]))))
        ccp4.set_extent(box)
        ccp4.write_ccp4_map(os.path.join(out_dir, fn))
        mobj = f"map{x.id}"
        lines.append(f"load {fn}, {mobj}")
        carve = f", carve={spec['carve']:g}" if spec.get("carve") else ""
        sel = " or ".join(sel_parts) or "all"
        cols = PALETTES[spec["palette"]]
        for k, lv in enumerate(levels):
            for sign in ((1, -1) if x.kind == "diff" else (1,)):
                absl = x.mean + sign * lv * x.rms
                obj = f"{mobj}_{'p' if sign > 0 else 'n'}{k}"
                col = (cols["pos"] if sign > 0 else cols["neg"]) if x.kind == "diff" else x.color
                lines += [f"isosurface {obj}, {mobj}, {absl:.6g}, ({sel}){carve}",
                          f"color {col.replace('#', '0x')}, {obj}",
                          f"set transparency, {1 - OPACITY[min(k, len(OPACITY) - 1)]:.2f}, {obj}"]
    # view: PyMOL's matrix is column-major model-to-camera rotation, then camera position, origin, clipping
    R = view_axes(spec["orient"])
    dist = half_h / np.tan(np.radians(FOV / 2))
    rot = [R[j, i] for i in range(3) for j in range(3)]
    view = rot + [0.0, 0.0, -dist] + list(c) + [dist - spec["slab"], dist + spec["slab"], 1.0]
    lines.append("set_view (" + ", ".join(f"{v:.6g}" for v in view) + ")")
    lines.append(f"# ray {spec['size_px'][0]}, {spec['size_px'][1]}")
    lines.append(f"# png {name}_pymol.png, dpi={spec['dpi']}")
    path = os.path.join(out_dir, name + ".pml")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path
