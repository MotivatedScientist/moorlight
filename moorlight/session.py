"""Session state and the command table shared by the UI, the HTTP API and the MCP tools."""
from __future__ import annotations

import asyncio
import base64
import glob
import inspect
import json
import os
import re
import threading
import time
import uuid

import gemmi
import numpy as np
from scipy.spatial import cKDTree

from . import edit
from .maps import XMap, detect_columns, check_columns, guess_kind, label_tells_kind, split_weight, DEFAULT_VISIBLE
from .model import BondOrders, Molecule

COMMANDS: dict[str, dict] = {}


def command(desc, params=None, view_only=False):
    """Register a Session method as a command. params: {name: (json_type, description)}."""
    def deco(fn):
        COMMANDS[fn.__name__] = {"fn": fn, "desc": desc, "params": params or {}, "view_only": view_only}
        return fn
    return deco


RES = {"chain": ("string", "chain id"), "resseq": ("integer", "residue number")}
RANGE = {"chain": ("string", "chain id"), "start": ("integer", "first residue number"),
         "end": ("integer", "last residue number (inclusive)")}
MOL = {"mol": ("integer", "molecule id (default: active molecule)")}
ATOM_SPEC = ("atom spec: 'A/85 CA', 'A/85 LYS CA:B' (altloc B), '2#A/85 SG' (molecule 2), 'A/85' (residue centroid), "
             "or {chain, resseq, atom, altloc, mol}")
MAPS = {"maps": ("array", "map ids (default: the visible maps)")}


def parse_atom_spec(spec):
    """'A/85 CA', 'A/85A LYS CA:B', '2#A/85/SG', 'A 85 CA' or a dict -> dict(mol, chain, resseq, icode, atom, altloc)."""
    if isinstance(spec, dict):
        d = dict(spec)
        d.setdefault("resseq", d.pop("res", None))
        return {"mol": d.get("mol"), "chain": str(d["chain"]), "resseq": int(d["resseq"]),
                "icode": (d.get("icode") or "").strip(), "atom": d.get("atom") or d.get("name"),
                "altloc": d.get("altloc") or None}
    t = str(spec).strip()
    mol = None
    m = re.match(r"^(\d+)\s*#\s*(.*)$", t)
    if m:
        mol, t = int(m.group(1)), m.group(2)
    if "/" in t:
        chain, rest = t.split("/", 1)
        toks = rest.replace("/", " ").split()
    else:
        toks = t.split()
        chain, toks = (toks[0], toks[1:]) if toks else ("", [])
    if not chain or not toks:
        raise ValueError(f"cannot read atom spec {spec!r}; use e.g. 'A/85 CA', 'A/85 LYS CA:B' or '2#A/85 SG'")
    m = re.match(r"^(-?\d+)([A-Za-z]?)$", toks[0])
    if not m:
        raise ValueError(f"bad residue number in atom spec {spec!r}")
    atom, alt = None, None
    rest = toks[1:]
    if rest:
        a = rest[-1]                       # [resname] atom[:altloc]
        if ":" in a:
            a, alt = a.split(":", 1)
        atom = a
    return {"mol": mol, "chain": chain.strip(), "resseq": int(m.group(1)), "icode": m.group(2),
            "atom": atom, "altloc": alt or None}


def _looks_like_difference_map(path, ccp4):
    """Guess whether a CCP4 map is a difference map: name hint, else mean ~ 0 with comparable positive and negative extremes."""
    import re
    base = os.path.basename(path).lower()
    if re.search(r"(^|[_\-.])(ded|diff|difference|fofo|fo-fo|fofc|mfo-dfc|delfwt|wdf|df)([_\-.]|$)", base) \
            and not re.search(r"(^|[_\-.])(2fofc|2mfo)", base):
        return True
    a = np.asarray(ccp4.grid)
    sd = float(a.std())
    if sd == 0:
        return False
    return abs(float(a.mean())) < 0.1 * sd and a.min() < -0.5 * a.max() and a.max() > 3.5 * sd and a.min() < -3.5 * sd


class Session:
    def __init__(self, workdir=None, sample_rate=3.0):
        self.workdir = workdir or os.getcwd()
        self.sample_rate = sample_rate
        self.molecules: dict[int, Molecule] = {}
        self.maps: dict[int, XMap] = {}
        self.active_mol: int | None = None
        self.refinement_map: int | None = None
        self.cif_paths: list[str] = []
        self.bond_orders = BondOrders(self.cif_paths)
        self.view = {"center": [0, 0, 0], "zoom": 30, "selection": [], "clicked": None}
        # map slices: style 'slice' draws maps as a plane (as in shared pages) instead of mesh contours. lock: the plane
        # fixed in the molecule {center, orient} (orient: view quaternion whose z row is the plane normal), else the
        # plane faces the viewer through the centre. map: map id shown (None: the first visible map).
        self.slice = {"style": "mesh", "map": None, "lock": None, "depth": 0.0, "thickness": 0.0}
        self.preview = None        # figure preview: {saved display state, note, radius} while on
        self.rsr = None
        self.rsr_thread = None
        self.rsr_weight = 60.0
        # refinement target: "map" (density of the refinement map), "difference" (model light-dark
        # difference density vs a difference map) or "both"
        self.rsr_target = "map"
        self.rsr_diff_weight = 60.0
        self.rsr_nonbonded = False  # atoms may be pulled on top of each other unless switched on
        self.diff_map: int | None = None
        self.dark_model: int | None = None  # None: other altlocs / start coordinates are the dark state
        self.log: list[str] = []
        self.lock = threading.RLock()
        self.broadcast = lambda msg: None  # set by server
        self.pending: dict[str, asyncio.Future] = {}
        self.loop: asyncio.AbstractEventLoop | None = None
        self.views: dict[str, dict] = self._read_json("views.json", {})
        self.figure_dir = self._read_json("settings.json", {}).get("figure_dir")   # chosen figure folder (None: <workdir>/figures)
        self.tour = {"title": "", "stops": []}
        self.tour_index = -1
        self.annotations = {"labels": [], "distances": []}   # atom specs set through the API (kept in views / tour stops)

    # ------------------------------------------------------------ plumbing
    def say(self, text):
        line = time.strftime("%H:%M:%S ") + text
        self.log.append(line)
        del self.log[:-300]
        self.broadcast({"type": "log", "text": line})

    def mol(self, mol=None) -> Molecule:
        mid = mol if mol is not None else self.active_mol
        if mid not in self.molecules:
            raise ValueError("no molecule loaded" if not self.molecules else f"no molecule {mid}")
        return self.molecules[mid]

    def figure_folder(self, out_dir=None, remember=True):
        """Folder for figures: out_dir (then remembered for this data, in .moorlight/settings.json), else the chosen
        one, else <workdir>/figures."""
        if out_dir:
            d = os.path.abspath(self._path(str(out_dir)))
            try:
                os.makedirs(d, exist_ok=True)
            except OSError as e:
                raise ValueError(f"cannot use {d} for figures: {e}") from None
            if remember and d != self.figure_dir:
                self.figure_dir = d
                st = self._read_json("settings.json", {})
                st["figure_dir"] = d
                self._write_json("settings.json", st)
                self.broadcast({"type": "figure_dir", "dir": d})
            return d
        return self.figure_dir or os.path.join(self.workdir, "figures")

    @command("Set the folder figures are written to (remembered for this data). Without path: back to <workdir>/figures.",
             {"path": ("string", "folder; ~ and paths relative to the work folder work")}, view_only=True)
    def set_figure_dir(self, path=None):
        if path:
            return {"figure_dir": self.figure_folder(path)}
        self.figure_dir = None
        st = self._read_json("settings.json", {})
        st.pop("figure_dir", None)
        self._write_json("settings.json", st)
        d = self.figure_folder()
        self.broadcast({"type": "figure_dir", "dir": d})
        return {"figure_dir": d}

    def _path(self, p):
        p = os.path.expanduser(p)
        return p if os.path.isabs(p) else os.path.join(self.workdir, p)

    def push_model(self, m: Molecule):
        self.broadcast({"type": "model", "mol": m.to_json()})

    def push_maps(self):
        self.broadcast({"type": "maps", "maps": [x.info() for x in self.maps.values()],
                        "refinement_map": self.refinement_map})

    def _require_no_rsr(self):
        if self.rsr is not None:
            raise RuntimeError("a refinement is in progress: accept (Enter) or reject (Esc) it first")

    def _edited(self, m, msg):
        self.push_model(m)
        self.say(msg)
        return msg

    # ------------------------------------------------------- atoms and maps
    def atoms(self, spec, mol=None, all_altlocs=True):
        """Resolve an atom spec -> (Molecule, [atom indices]). A residue spec gives its heavy atoms. Without an altloc all
        conformers match (all_altlocs) or only the first one (blank or lowest altloc)."""
        d = parse_atom_spec(spec)
        m = self.mol(d["mol"] if d["mol"] is not None else mol)
        res = [i for i, k in enumerate(m.keys) if k[0] == d["chain"] and k[1] == d["resseq"] and k[2] == d["icode"]]
        if d["atom"] and res and d["atom"] == m.info[res[0]][3] and not any(m.keys[i][3] == d["atom"] for i in res):
            d["atom"] = None                  # 'A/85 LYS': a residue name, not an atom
        idx = [i for i in res if (m.keys[i][3] == d["atom"] if d["atom"] else m.elements[i] != "H")
               and (d["altloc"] is None or m.keys[i][4] in (d["altloc"], ""))]
        if not idx:
            raise ValueError(f"no atom matches {spec!r} in molecule {m.id} ({m.name})")
        if d["altloc"] is None and not all_altlocs:
            alts = sorted({m.keys[i][4] for i in idx} - {""})
            idx = [i for i in idx if m.keys[i][4] in ("", alts[0] if alts else "")]
        return m, idx

    def point(self, spec, mol=None):
        """Coordinates of an atom spec (first conformer) or residue centroid; a list of specs gives their centroid."""
        if isinstance(spec, (list, tuple)) and spec and not isinstance(spec[0], (int, float)):
            return np.mean([self.point(s, mol) for s in spec], axis=0)
        if isinstance(spec, (list, tuple)):
            return np.asarray(spec, float)
        m, idx = self.atoms(spec, mol, all_altlocs=False)
        return m.xyz[idx].mean(0)

    def map_list(self, maps=None):
        """Map ids (list, '3,4' or single) -> XMap list; default: visible maps (all maps if none is visible)."""
        if maps is None or maps == []:
            out = [x for x in self.maps.values() if x.visible]
            return out or list(self.maps.values())
        if isinstance(maps, (str, int)):
            maps = str(maps).replace(",", " ").split()
        out = []
        for i in maps:
            if int(i) not in self.maps:
                raise ValueError(f"no map {i}; maps: {sorted(self.maps)}")
            out.append(self.maps[int(i)])
        return out

    @staticmethod
    def sigma(x, xyz):
        return round(float((x.value_at(xyz) - x.mean) / x.rms), 2)

    @staticmethod
    def map_key(x):
        return f"[{x.id}] {x.name}"

    def _read_json(self, name, default):
        try:
            with open(os.path.join(self.workdir, ".moorlight", name)) as fh:
                return json.load(fh)
        except Exception:
            return default

    def _write_json(self, name, data):
        d = os.path.join(self.workdir, ".moorlight")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, name), "w") as fh:
            json.dump(data, fh, indent=1)

    def run(self, name, args):
        """Execute a command synchronously (called in a worker thread)."""
        if name not in COMMANDS:
            raise ValueError(f"unknown command {name}")
        spec = COMMANDS[name]
        fn = spec["fn"]
        sig = inspect.signature(fn)
        clean = {k: v for k, v in (args or {}).items() if k in sig.parameters and v is not None}
        with self.lock:
            return fn(self, **clean)

    # ----------------------------------------------------------- awaiting UI
    def ask_client(self, kind, payload, timeout=10):
        """Request something from the browser (e.g. a screenshot) and wait for the reply."""
        if self.loop is None:
            raise RuntimeError("no event loop")
        rid = uuid.uuid4().hex
        fut = asyncio.run_coroutine_threadsafe(self._ask(rid, kind, payload, timeout), self.loop)
        return fut.result(timeout + 1)

    async def _ask(self, rid, kind, payload, timeout):
        f = self.loop.create_future()
        self.pending[rid] = f
        self.broadcast({"type": kind, "id": rid, **payload})
        try:
            return await asyncio.wait_for(f, timeout)
        finally:
            self.pending.pop(rid, None)

    def reply(self, rid, data):
        f = self.pending.get(rid)
        if f and not f.done():
            f.set_result(data)

    # ================================================================ state
    @command("Describe what the user currently sees: molecules, maps (visible, contour level), view centre "
             "and nearest residue, selected atoms, refinement in progress, fixed atoms, recent log. compact=true gives one line "
             "per molecule and map.", {"compact": ("boolean", "one line per molecule / map")}, view_only=True)
    def state(self, compact=False):
        if compact:
            v = self.view
            return {"molecules": [f"{m.id}: {m.name} {'visible' if m.visible else 'hidden'}{' ACTIVE' if m.id == self.active_mol else ''}"
                                  f" {len(m.keys)} atoms" for m in self.molecules.values()],
                    "maps": [f"{x.id}: {x.name} | {x.source} | {x.kind} | {'visible' if x.visible else 'hidden'} | "
                             f"{x.level_sigma:g}σ{' | R' if x.id == self.refinement_map else ''}" for x in self.maps.values()],
                    "view": {k: v.get(k) for k in ("center", "zoom", "slab")},
                    "views": list(self.views), "tour": f"{self.tour_index + 1}/{len(self.tour['stops'])}" if self.tour["stops"] else None,
                    "slice": self.slice if self.slice["style"] == "slice" else None}
        out = {"workdir": self.workdir, "molecules": [], "maps": [], "view": {}}
        for m in self.molecules.values():
            chains = [(ch.name, len(ch)) for ch in m.st[0]]
            out["molecules"].append({"id": m.id, "name": m.name, "path": m.path, "atoms": len(m.keys),
                                     "visible": m.visible, "active": m.id == self.active_mol,
                                     "chains": chains, "fixed_atoms": len(m.fixed),
                                     "spacegroup": m.st.spacegroup_hm, "undo_depth": len(m.undo_stack)})
        for x in self.maps.values():
            i = x.info()
            i["refinement_map"] = x.id == self.refinement_map
            out["maps"].append(i)
        v = dict(self.view)
        c = v.get("center")
        if c and self.molecules and self.active_mol in self.molecules:
            m = self.mol()
            if len(m.xyz):
                d, i = cKDTree(m.xyz).query(c)
                k = m.info[i]
                v["nearest_atom"] = f"{k[0]}/{k[1]}{k[2]} {k[3]} {k[4]}{(':' + k[5]) if k[5] else ''} ({d:.1f} Å)"
        sel = v.get("selection") or []
        if sel and self.active_mol in self.molecules:
            m = self.mol()
            v["selection_atoms"] = [self._atom_label(m, i) for i in sel if i < len(m.info)]
        out["view"] = v
        out["refinement_settings"] = self.rsr_settings()
        out["slice"] = self.slice
        out["figure_dir"] = self.figure_folder()
        if self.rsr is not None:
            out["refinement"] = {"running": True, "moving_atoms": len(self.rsr.moving_mol),
                                 "altloc": self.rsr.altloc, **self.rsr.geometry_stats()}
        out["log"] = self.log[-15:]
        out["views"] = list(self.views)
        if self.tour["stops"]:
            out["tour"] = {"title": self.tour["title"], "stops": len(self.tour["stops"]), "current": self.tour_index}
        return out

    @staticmethod
    def _atom_label(m, i):
        k = m.info[i]
        return f"{k[0]}/{k[1]}{k[2]} {k[3]} {k[4]}{(':' + k[5]) if k[5] else ''}"

    @command("Take a screenshot of the 3D view, with atom labels and distances drawn in. Returns the path of a PNG file you can "
             "Read. By default it waits until the view has stopped moving and the map contours for it are computed.",
             {"width": ("integer", "output width in pixels (default: as on screen, at most 1000)"),
              "crop": ("array", "[x, y, w, h] as fractions of the view, e.g. [0.2, 0.1, 0.6, 0.8]"),
              "wait": ("boolean", "wait for pending map contours (default true)"),
              "labels": ("boolean", "draw labels and distances (default true)"),
              "path": ("string", "output PNG path (default .moorlight/screenshots/)")}, view_only=True)
    def screenshot(self, width=None, crop=None, wait=True, labels=True, path=None):
        data = self.ask_client("screenshot_request", {"width": width, "crop": crop, "wait": wait, "labels": labels},
                               timeout=120 if wait else 15)
        if not isinstance(data, str) or "," not in data:
            raise RuntimeError(f"screenshot failed: {data}")
        png = base64.b64decode(data.split(",", 1)[1])
        d = os.path.join(self.workdir, ".moorlight", "screenshots")
        os.makedirs(d, exist_ok=True)
        p = self._path(path) if path else os.path.join(d, time.strftime("view_%Y%m%d_%H%M%S.png"))
        with open(p, "wb") as fh:
            fh.write(png)
        return p

    # ============================================================= figures
    @command("Publication figure of the current view: maps as nested transparent surfaces (weaker levels more "
             "transparent), cut to the shown atoms (carve), on white, orthographic, at journal size. Writes <name>.png "
             "(labels drawn in), <name>_plain.png, <name>.svg (labels as editable text), <name>_uncarved.png, "
             "<name>.json (to make it again) and <name>_legend.md (draft legend with levels and carving); "
             "pymol=true adds a PyMOL script with the same view. Returns the paths and warnings, e.g. density that "
             "carving hides.",
             {"name": ("string", "file name stem (default: figure_<time>, or the tour stop's name)"),
              "preset": ("string", "single (89 mm, default), one-and-half (120 mm), double (183 mm), slide (16:9)"),
              "width_mm": ("number", "width in mm (overrides the preset)"),
              "aspect": ("number", "width / height (default: as on screen; slide: 16/9)"),
              "dpi": ("integer", "dots per inch (default 300)"),
              "background": ("string", "white (default), black or transparent"),
              "style": ("string", "surfaces (default), slice (the current slice plane) or none (no maps)"),
              "maps": ("array", "map ids (default: the visible maps)"),
              "levels": ("array", "sigma levels for every map (default: each map's level x 1, 1.5, 2.25)"),
              "carve": ("number", "show density within this many Å of the shown atoms (default 2; 0: no carving)"),
              "uncarved": ("boolean", "also write the uncarved image (default true when carving)"),
              "palette": ("string", "standard (green/red) or cvd (blue/orange, readable with red-green colour blindness)"),
              "font_pt": ("number", "label size in points (default 7)"),
              "title": ("string", "figure title for the legend"),
              "text": ("string", "what the figure shows, for the legend (default: the tour stop's text)"),
              "out_dir": ("string", "folder (default: the chosen figure folder, else <workdir>/figures); remembered"),
              "pymol": ("boolean", "also write a PyMOL script and cut maps"),
              "look": ("boolean", "figure look (default true): only for the render, main chain of stick models as "
                                  "cartoon and only residues near the labelled atoms (or the centre); off: as on screen"),
              "look_radius": ("number", "Å around the labelled atoms for the figure look (default 5)"),
              "opacity": ("array", "surface opacities, outer (weakest level) to inner, e.g. [0.1, 0.2, 0.3] (default: "
                                   "density maps 0.12, 0.2, 0.32; difference maps 0.22, 0.38, 0.6)")}, view_only=True)
    def figure(self, name=None, preset="single", width_mm=None, aspect=None, dpi=300, background="white",
               style="surfaces", maps=None, levels=None, carve=2.0, uncarved=None, palette="standard", font_pt=7.0,
               title=None, text=None, out_dir=None, pymol=False, look=True, look_radius=5.0, remember=True, opacity=None):
        """look: only while rendering, draw the main chain of stick models as cartoon and only residues within
        look_radius of the labelled atoms (or of the centre); the view is restored afterwards."""
        if self.preview is not None:          # figure preview on: the look is already applied
            return self._figure(name, preset, width_mm, aspect, dpi, background, style, maps, levels, carve, uncarved,
                                palette, font_pt, title, text, out_dir, pymol, self.preview["note"], remember, opacity)
        saved, note = self._apply_look(look_radius) if look else ({}, None)
        try:
            return self._figure(name, preset, width_mm, aspect, dpi, background, style, maps, levels, carve, uncarved,
                                palette, font_pt, title, text, out_dir, pymol, note, remember, opacity)
        finally:
            self._restore_look(saved)

    def _apply_look(self, radius):
        """Figure look: stick models get a cartoon main chain, only residues within `radius` of the labelled atoms (or
        of the centre) are drawn. Returns (saved display state, note for the legend)."""
        focus, around = [], "labelled atoms"
        for mid, i in self.view.get("labelled") or []:          # atoms labelled by clicking in the viewer
            m = self.molecules.get(int(mid))
            if m is not None and 0 <= int(i) < len(m.xyz):
                focus.append(m.xyz[int(i)])
        for spec in self.annotations.get("labels", []) + [x for d in self.annotations.get("distances", []) for x in d[:2]]:
            try:
                focus.append(self.point(spec))
            except Exception:
                pass
        if not focus:
            focus, around = [np.asarray(self.view.get("center") or [0, 0, 0], float)], "centre of the view"
        tree = cKDTree(np.array(focus))
        saved, cartooned = {}, []
        for m in self.molecules.values():
            if not m.visible or not len(m.xyz):
                continue
            saved[m.id] = (m.style, m.zone)
            if m.zone is None:
                d = tree.query(m.xyz)[0]
                m.zone = sorted({k[:3] for k, dd in zip(m.keys, d) if dd <= float(radius)})
            if (m.style or "sticks") == "sticks" and any(i[4] == "CA" and not i[8] for i in m.info):
                m.style = "cartoon"
                cartooned.append(m.name)
            self.push_display(m)
        return saved, {"cartoon": cartooned, "radius": float(radius), "around": around}

    def _restore_look(self, saved):
        for mid, (st, zn) in saved.items():
            m = self.molecules.get(mid)
            if m is not None:
                m.style, m.zone = st, zn
                self.push_display(m)

    @command("Figure preview in the viewer: shows what a figure will show (figure look: cartoon main chain, only "
             "residues near the labelled atoms; white background, figure atoms, placed labels). Maps stay as mesh; "
             "they become surfaces in the figure. on=true again refreshes the zone after new labels.",
             {"on": ("boolean", "preview on (default true) or off"),
              "look_radius": ("number", "Å around the labelled atoms (default 5)")}, view_only=True)
    def figure_preview(self, on=True, look_radius=5.0):
        if self.preview is not None:
            self._restore_look(self.preview["saved"])
            self.preview = None
        if on:
            saved, note = self._apply_look(look_radius)
            self.preview = {"saved": saved, "note": note, "radius": float(look_radius)}
        self.broadcast({"type": "figure_preview", "on": bool(on), "radius": float(look_radius)})
        return {"preview": bool(on), **({"look": self.preview["note"]} if on else {})}

    def _figure(self, name, preset, width_mm, aspect, dpi, background, style, maps, levels, carve, uncarved, palette,
                font_pt, title, text, out_dir, pymol, look_note, remember=True, opacity=None):
        from . import figure as F
        if preset not in F.PRESETS:
            raise ValueError(f"preset: one of {list(F.PRESETS)}")
        if style not in ("surfaces", "slice", "none"):
            raise ValueError("style: surfaces, slice or none")
        if background not in ("white", "black", "transparent"):
            raise ValueError("background: white, black or transparent")
        if palette not in F.PALETTES:
            raise ValueError(f"palette: one of {list(F.PALETTES)}")
        try:
            v = self.ask_client("view_request", {}, timeout=5)
        except Exception as e:
            raise RuntimeError(f"figure needs the viewer open in a browser tab ({e})") from None
        self.view.update({k: v[k] for k in ("center", "orient", "zoom", "slab") if k in v})
        screen_aspect = float(v.get("aspect") or 1.0)
        w_mm = float(width_mm or F.PRESETS[preset][0])
        asp = float(aspect or F.PRESETS[preset][1] or min(2.5, max(0.5, screen_aspect)))
        dpi = int(dpi)
        W = int(round(w_mm / 25.4 * dpi))
        H = int(round(W / asp))
        center = np.asarray(v["center"], dtype=float)
        axes = F.view_axes(v.get("orient"))
        slab = float(v.get("slab") or 10)
        half_w, half_h = F.frame(v, W / H, screen_aspect)
        atoms, _ = F.shown_atoms(self, center, axes, half_w, half_h, slab)
        carve = float(carve or 0)
        if uncarved is None:
            uncarved = carve > 0
        stop = self.tour["stops"][self.tour_index] if self.tour["stops"] and 0 <= self.tour_index < len(self.tour["stops"]) else None
        at_stop = stop is not None and np.linalg.norm(np.asarray(stop["view"].get("center") or [1e9] * 3) - center) < 0.5
        if text is None and at_stop:
            text = stop.get("text", "")
        if title is None and at_stop:
            title = stop.get("name", "")
        name = F.slug(name or (title if at_stop else None) or time.strftime("figure_%Y%m%d_%H%M%S"))
        xs = [] if style != "surfaces" else self.map_list(maps) if maps else [x for x in self.maps.values() if x.visible]
        map_levels = []
        for x in xs:
            lv = [float(l) for l in levels] if levels else [round(x.level_sigma * k, 2) for k in F.LEVEL_STEPS]
            map_levels.append((x, lv))
        radius = float(min(25.0, max(half_w, half_h, slab) + 1.5))
        self._figure_job = {"center": center.tolist(), "radius": radius, "carve_xyz": atoms, "carve": carve,
                            "levels": {x.id: lv for x, lv in map_levels}}
        cols = F.PALETTES[palette]
        payload = {"width": W, "height": H, "background": background, "style": style, "half_w": half_w,
                   "half_h": half_h, "font_px": font_pt / 72 * dpi, "uncarved": bool(uncarved and carve > 0 and map_levels),
                   "maps": [{"id": x.id, "kind": x.kind, "levels": lv,
                             "colors": [cols["pos"], cols["neg"]] if x.kind == "diff" else [x.color],
                             "opacity": F.opacities(x.kind, len(lv), opacity)} for x, lv in map_levels]}
        res = self.ask_client("figure_request", payload, timeout=300)
        if not isinstance(res, dict) or "png" not in res:
            raise RuntimeError(f"figure failed: {res}")
        carbon = res.get("carbon", {})
        spec = {"name": name, "title": title or "", "text": text or "", "preset": preset, "size_mm": [w_mm, w_mm / asp],
                "size_px": [W, H], "dpi": dpi, "background": background, "style": style, "palette": palette,
                "carve": carve, "uncarved": bool(payload["uncarved"]), "opacity": opacity, "center": center.tolist(), "orient": v.get("orient"),
                "zoom": v.get("zoom"), "slab": slab, "carbon": carbon,
                "maps": [{"id": x.id, "name": x.name, "kind": x.kind, "levels": lv, "source": x.source,
                          "f": x.f, "phi": x.phi} for x, lv in map_levels],
                "models": [{"id": m.id, "name": m.name, "path": m.path} for m in self.molecules.values() if m.visible],
                "distances": bool(any(lb.get("kind") == "distance" for lb in res.get("labels", []))),
                "hbonds": bool(res.get("hbonds")), "coordination": bool(res.get("coordination")),
                "look": look_note,
                "view": self._snapshot()}
        d = self.figure_folder(out_dir, remember=remember)
        paths = F.write_files(d, name, {**res, "width": W, "height": H}, spec)
        maps_info = [{"name": x.name, "kind": x.kind, "levels": lv, "source": x.source, "rms": x.rms,
                      "resolution": round(x.resolution, 2) if x.resolution else None} for x, lv in map_levels]
        models_info = [{"name": m.name, "carbon": carbon.get(str(m.id), "?")} for m in self.molecules.values() if m.visible]
        with open(os.path.join(d, name + "_legend.md"), "w") as fh:
            fh.write(F.legend(spec, maps_info, models_info))
        paths["legend"] = os.path.join(d, name + "_legend.md")
        warnings = []
        if carve > 0:
            for x, lv in map_levels:
                hid = F.hidden_by_carving(x, center, axes, half_w, half_h, slab, atoms, carve)
                if hid and abs(hid[0]) >= min(lv):
                    see = f"see {name}_uncarved.png" if payload["uncarved"] else "make it with uncarved=true to see it"
                    warnings.append(f"map {x.name!r}: carving hides {hid[0]:+.1f} σ at {hid[1]}, more than {carve:g} Å "
                                    f"from the shown atoms ({see})")
        if pymol:
            paths["pymol"] = F.pymol_script(self, d, name, spec, map_levels, half_w, half_h)
        self.say(f"figure {name}: {W}×{H} px ({w_mm:g} mm at {dpi} dpi) in {d}")
        return {"files": paths, "size_px": [W, H], "size_mm": [round(w_mm, 1), round(w_mm / asp, 1)],
                "maps": [{"name": x.name, "levels": lv} for x, lv in map_levels], "warnings": warnings}

    @command("Publication figures of every stop of the current tour (see figure): one set of files per stop, named "
             "after the stop, with its text in the legend, plus figures.md with all legends in order. Takes the "
             "options of figure.",
             {"out_dir": ("string", "folder; the figures go to <folder>/<tour title> (default: the figure folder); remembered"),
              "options": ("object", "options passed to figure for every stop (preset, dpi, carve, ...)")}, view_only=True)
    def figure_tour(self, out_dir=None, options=None):
        from . import figure as F
        if not self.tour["stops"]:
            raise ValueError("no tour loaded (tour_load) or built (tour_add)")
        d = os.path.join(self.figure_folder(out_dir), F.slug(self.tour["title"], "tour"))
        opts = {k: v for k, v in (options or {}).items() if k not in ("out_dir", "name", "title", "text")}
        prev = self.preview
        if prev is not None:                  # each stop applies the look around its own labels
            self.figure_preview(on=False)
            opts.setdefault("look_radius", prev["radius"])
        out, legends = [], [f"# {self.tour['title'] or 'Figures'}\n"]
        for i, st in enumerate(self.tour["stops"]):
            self.tour_goto(index=i)
            time.sleep(0.6)
            nm = f"fig{i + 1:02d}_{F.slug(st.get('name'), 'stop')}"
            r = self.figure(name=nm, title=st.get("name", ""), text=st.get("text", ""), out_dir=d, remember=False, **opts)
            out.append({"stop": st.get("name"), **r})
            with open(r["files"]["legend"]) as fh:
                legends.append(f"## Figure {i + 1}: {st.get('name', '')}\n\n" + fh.read())
        with open(os.path.join(d, "figures.md"), "w") as fh:
            fh.write("\n".join(legends))
        if prev is not None:
            self.figure_preview(on=True, look_radius=prev["radius"])
        return {"folder": d, "figures": out}

    # ============================================================== loading
    @command("Load a model (PDB or mmCIF).", {"path": ("string", "file path")})
    def load_model(self, path):
        self._require_no_rsr()
        p = self._path(path)
        m = Molecule(p, bond_orders=self.bond_orders)
        self.molecules[m.id] = m
        self.active_mol = m.id
        self.push_model(m)
        self.say(f"loaded model {m.name} ({len(m.keys)} atoms) as molecule {m.id}")
        self._import_dictionaries(p)
        if len(self.molecules) == 1 and len(m.xyz):
            self.set_center(*m.xyz.mean(0).tolist())
        return {"mol": m.id, "atoms": len(m.keys)}

    def _import_dictionaries(self, model_path):
        """Pick up ligand/modified-residue dictionaries embedded in refinement output mmCIF
        (the model itself, or phenix's <stem>.cif written next to <stem>.pdb)."""
        from . import dicts

        stem = os.path.splitext(model_path)[0]
        have = {os.path.splitext(os.path.basename(p))[0] for p in self.cif_paths}
        for src in dict.fromkeys([model_path, stem + ".cif"]):
            if not (src.lower().endswith(".cif") and os.path.exists(src)) or not dicts.comp_ids(src):
                continue
            out = os.path.join(self.workdir, ".moorlight", "dicts", os.path.basename(stem))
            try:
                paths = dicts.extract_all(src, out)
            except Exception as e:
                self.say(f"could not read dictionaries from {os.path.basename(src)}: {e}")
                continue
            new = [p for p in paths if os.path.splitext(os.path.basename(p))[0] not in have]
            self.cif_paths += new
            if new:
                self.say("restraint dictionaries from " + os.path.basename(src) + ": " +
                         ", ".join(os.path.splitext(os.path.basename(p))[0] for p in new))

    @command("Load maps from an MTZ. TorchRef (FWT/PHWT, DELFWT/PHDELWT) and TorchRef difference-refinement "
             "(WDF, DF, DFc with PHIC_dark, mDFop-DFc/PHIC_diff, extrapolated maps) columns are detected "
             "automatically, as are phenix maps (2FOFCWT/FOFCWT, Fo-Fo FoFo/PHFc, polder/omit). Optionally give explicit columns as "
             "'F,PHI', 'F,PHI,kind' or 'F,PHI,kind,name' strings (kind 'diff' or '2fofc'). label names the maps (one map: the "
             "name; several: a prefix). d_min truncates the data, to compare data sets at matched resolution.",
             {"path": ("string", "MTZ path"),
              "columns": ("array", "optional list of 'F,PHI[,kind[,name]]' strings"),
              "label": ("string", "map name (or prefix if several maps are loaded), e.g. 'pH 7 2mFo-DFc'"),
              "d_min": ("number", "resolution cut-off in Å (default: all data)"),
              "show_all": ("boolean", "make all detected maps visible")})
    def load_mtz(self, path, columns=None, show_all=False, label=None, d_min=None):
        p = self._path(path)
        mtz = gemmi.read_mtz_file(p)
        open_kind = set()          # F labels whose kind is only a guess (decided from the values when computed)
        if columns:
            found = []
            if isinstance(columns, str):
                columns = [columns]
            for c in columns:
                parts = [s.strip() for s in c.split(",")]
                if len(parts) < 2:
                    raise ValueError(f"columns entry {c!r}: expected 'F,PHI[,kind[,name]]'")
                check_columns(mtz, parts[0], parts[1])
                kind = parts[2] if len(parts) > 2 and parts[2] else guess_kind(parts[0])
                if not (len(parts) > 2 and parts[2]) and not label_tells_kind(parts[0]):
                    open_kind.add(parts[0])
                if kind not in ("diff", "2fofc"):
                    raise ValueError(f"kind must be 'diff' or '2fofc', not {kind!r}")
                found.append((parts[0], parts[1], kind, parts[3] if len(parts) > 3 else f"{parts[0]}/{parts[1]}"))
        else:
            found = detect_columns(mtz)
            open_kind = {f for f, phi, kind, name in found if kind == "2fofc" and f == name.split("/")[0]
                         and not label_tells_kind(f)}
        if label:
            found = [(f, phi, kind, label if len(found) == 1 else f"{label} {name}") for f, phi, kind, name in found]
        if d_min:
            d_min = float(d_min)
            found = [(f, phi, kind, f"{name} (d_min {d_min:g} Å)") for f, phi, kind, name in found]
        if not found:
            raise ValueError(f"no map coefficient columns recognised in {p}: "
                             f"{[c.label for c in mtz.columns]}")
        new = []
        first_vis = not self.maps
        for f, phi, kind, label in found:
            vis = show_all or (first_vis and f in DEFAULT_VISIBLE)
            x = XMap.from_mtz(mtz, f, phi, kind, label, os.path.basename(p), self.sample_rate, lazy=not vis, d_min=d_min)
            x.path = p
            x.visible = vis
            if f in open_kind:
                x.kind_open = True
                if x.computed:                  # already computed (visible): decide now
                    x.kind_open = False
                    from .maps import kind_from_values
                    k = kind_from_values(np.array(x.grid, copy=False))
                    if k != x.kind:
                        from .maps import COLORS, DEFAULT_LEVEL
                        x.kind, x.level_sigma, x.color = k, DEFAULT_LEVEL[k], COLORS[k]
            self.maps[x.id] = x
            new.append(x)
            if self.refinement_map is None and kind == "2fofc" and f not in ("F-model", "Fc_dark", "Fc_light"):
                self.refinement_map = x.id
            if self.diff_map is None and split_weight(f)[0] in ("WDF", "DF", "dF"):
                self.diff_map = x.id
        self.push_maps()
        self.say(f"{os.path.basename(p)}: " + ", ".join(f"[{x.id}] {x.name}" for x in new))
        return [x.info() for x in new]

    @command("Load a CCP4/MRC map file.", {"path": ("string", "map path"),
                                            "difference": ("boolean", "contour at +/- level; omit to auto-detect "
                                                                  "(file name or symmetric +/- density)"),
                                            "label": ("string", "map name (default: file name)")})
    def load_map(self, path, difference=None, label=None):
        p = self._path(path)
        ccp4 = gemmi.read_ccp4_map(p, setup=True)
        if difference is None:
            difference = _looks_like_difference_map(p, ccp4)
        x = XMap(label or os.path.basename(p), ccp4.grid, "diff" if difference else "2fofc", os.path.basename(p))
        x.path = p
        x.visible = True
        self.maps[x.id] = x
        self.push_maps()
        return x.info()

    @command("Open a TorchRef result directory: loads refined.pdb + refined.mtz, or for difference "
             "refinement D.pdb, L_refined.pdb and difference_data.mtz.", {"path": ("string", "directory")})
    def load_torchref(self, path):
        d = self._path(path)
        names = set(os.listdir(d))
        models = [n for n in ("D.pdb", "L_refined.pdb", "refined.pdb") if n in names]
        pre = next((n[:-len("difference_data.mtz")] for n in sorted(names) if n.endswith("difference_data.mtz")), None)
        if not models and pre is not None and pre + "dark.pdb" in names and pre + "light.pdb" in names:
            models = [pre + "dark.pdb", pre + "light.pdb"]      # current TorchRef layout: <prefix>_dark/_light.pdb
        if not models:
            # e.g. <prefix>_refined.pdb; fall back to the only model in the folder
            models = sorted(n for n in names if n.endswith("refined.pdb")) or \
                sorted(n for n in names if n.endswith(".pdb"))[:1]
        if not models:
            models = sorted(n for n in names if n.endswith(("refined.cif", ".mmcif")))[:1]
        mtzs = [n for n in ("refined.mtz", "difference_data.mtz") if n in names] or \
            sorted(n for n in names if n.endswith(".mtz"))
        if pre is not None and pre + "difference_data.mtz" in mtzs:   # difference maps first: shown, and their extrapolated
            mtzs.remove(pre + "difference_data.mtz")                     # light map is the refinement map for the light model
            mtzs.insert(0, pre + "difference_data.mtz")
        # dictionaries: CIFs in the folder, plus the ones named in TorchRef logs ("CIF file: ...")
        cifs = [os.path.join(d, n) for n in sorted(names) if n.endswith(".cif") and "refined" not in n]
        for log in sorted(n for n in names if n.endswith(".log")):
            try:
                with open(os.path.join(d, log), errors="ignore") as fh:
                    for line in fh:
                        if "CIF file:" in line:
                            cifs.append(line.split("CIF file:", 1)[1].strip())
            except OSError:
                pass
        for f in dict.fromkeys(cifs):
            if f in self.cif_paths or not os.path.exists(f):
                continue
            try:
                if any(b.name.startswith("comp_") for b in gemmi.cif.read(f)):
                    self.cif_paths.append(f)
                    self.say(f"restraint dictionary {f}")
            except Exception:
                pass
        ids = {n: self.load_model(os.path.join(d, n))["mol"] for n in models}
        for n in mtzs:
            self.load_mtz(os.path.join(d, n))
        pair = next(((d, l) for d, l in (("D.pdb", "L_refined.pdb"), (f"{pre}dark.pdb", f"{pre}light.pdb"))
                     if d in ids and l in ids), None)
        if pair:
            self.dark_model = ids[pair[0]]
            self.set_active_molecule(ids[pair[1]])
            self.say(f"difference refinement: edits go to {pair[1]}, {pair[0]} is the dark reference")
        self.push_rsr_settings()
        if not models:
            raise ValueError(f"no models found in {d}")
        return self.state()

    @command("Re-read a molecule and all maps from disk (e.g. after a refinement run finished).",
             {"mol": MOL["mol"], "maps": ("boolean", "also reload maps (default true)")})
    def reload(self, mol=None, maps=True):
        self._require_no_rsr()
        m = self.mol(mol)
        m.checkpoint()
        m.st = gemmi.read_structure(m.path)
        m.rebuild()
        self.push_model(m)
        if maps:
            by_src = {}
            for x in self.maps.values():
                by_src.setdefault(x.source, []).append(x)
            for src, xs in by_src.items():
                path = next((os.path.join(os.path.dirname(mm.path), src) for mm in self.molecules.values()
                             if os.path.exists(os.path.join(os.path.dirname(mm.path), src))), None)
                if not path:
                    continue
                mtz = gemmi.read_mtz_file(path)
                for x in xs:
                    if x.f:
                        new = XMap.from_mtz(mtz, x.f, x.phi, x.kind, x.name, src, self.sample_rate, d_min=x.d_min)
                        x.grid, x.mean, x.rms = new.grid, new.mean, new.rms
            self.push_maps()
            self.broadcast({"type": "maps_changed"})
        self.say(f"reloaded {m.name}")
        return "ok"

    @command("Register a restraint dictionary (CIF) for ligands / modified residues used in refinement.",
             {"path": ("string", "CIF path")})
    def add_dictionary(self, path):
        p = self._path(path)
        if p not in self.cif_paths:
            self.cif_paths.append(p)
        return self.cif_paths

    @command("Save a molecule to PDB or mmCIF.", {"path": ("string", "output path (.pdb or .cif)"), **MOL})
    def save_model(self, path, mol=None):
        m = self.mol(mol)
        p = m.write(self._path(path))
        self.say(f"saved {m.name} -> {p}")
        return p

    # ================================================================= view
    @command("Centre the view on a residue and frame it: centred on the side chain (whole residue for waters/ions), zoomed so the "
             "side chain plus `radius` Å of context fits the screen, with a depth slab of +/- `slab` Å so only the relevant atoms are "
             "drawn (default 2.8 Å; the framed radius is capped at 9 Å for large residues). The view is rotated so the long axis of the "
             "side chain is horizontal (CA on the left) and its flat face looks at the viewer (orient=false keeps the current rotation). "
             "radius=0 only centres the view (on CA; zoom, slab and rotation unchanged).",
             {**RES, "icode": ("string", "insertion code"), **MOL,
              "radius": ("number", "Å of context around the side chain (default 4.5)"),
              "slab": ("number", "half-thickness in depth, Å (default 2.8)"),
              "orient": ("boolean", "rotate the view: long axis of the side chain horizontal (default true)"),
              "roll": ("number", "extra roll about the viewing axis in degrees (default 0)")}, view_only=True)
    def goto(self, chain, resseq, icode="", mol=None, radius=4.5, slab=None, orient=True, roll=0.0):
        m = self.mol(mol)
        if not radius:
            c = m.residue_center(chain, int(resseq), icode)
            self.set_center(*c)
            return c
        c, extent = m.residue_extent(chain, int(resseq), icode)
        user_slab = float(slab) if slab else None
        fit = min(extent + float(radius), max(9.0, float(radius)))   # large modified residues: do not zoom out beyond ~9 Å
        slab = user_slab or 2.8
        if extent < 0.1:   # single atom (water, ion): include its hydrogen-bond / coordination partners
            reach = m.polar_reach(c, 3.4, exclude_key=(chain, int(resseq), (icode or "").strip())) or 2.8
            fit = max(6.0, reach + 4.1)
            slab = user_slab or reach + 0.6
        self.view["center"] = c
        q = m.residue_orientation(chain, int(resseq), icode, float(roll)) if orient else None
        self.broadcast({"type": "view_cmd", "center": c, "fit_radius": fit, "slab": slab, **({"orient": q} if q else {})})
        return {"center": [round(v, 3) for v in c], "fit_radius": round(fit, 2), "slab": round(slab, 2), "oriented": q is not None}

    @command("Centre the view on coordinates, on an atom, or on the centroid of several atoms (e.g. the midpoint of S and C: "
             "atom=['A/85 SG', 'B/501 C1']). Zoom, slab and rotation are kept.",
             {"x": ("number", ""), "y": ("number", ""), "z": ("number", ""),
              "atom": ("array", "atom spec or list of atom specs; " + ATOM_SPEC), **MOL}, view_only=True)
    def set_center(self, x=None, y=None, z=None, atom=None, mol=None):
        if atom is not None:
            x, y, z = (float(v) for v in self.point(atom if isinstance(atom, list) else [atom], mol))
        if x is None or y is None or z is None:
            raise ValueError("give x, y, z or atom")
        x, y, z = float(x), float(y), float(z)
        self.view["center"] = [x, y, z]
        self.broadcast({"type": "view_cmd", "center": [x, y, z]})
        return [round(x, 3), round(y, 3), round(z, 3)]

    @command("Centre on one atom and zoom so `radius` Å around it fits the screen; the rotation is kept.",
             {"atom": ("string", ATOM_SPEC), **MOL, "radius": ("number", "Å of context (default 6)"),
              "slab": ("number", "depth half-thickness in Å (default 4)")}, view_only=True)
    def goto_atom(self, atom, mol=None, radius=6.0, slab=4.0):
        m, idx = self.atoms(atom, mol, all_altlocs=False)
        c = m.xyz[idx[0]].tolist()
        self.view["center"] = c
        self.broadcast({"type": "view_cmd", "center": c, "fit_radius": float(radius), "slab": float(slab)})
        return {"atom": self._atom_label(m, idx[0]), "center": [round(v, 3) for v in c]}

    @command("Frame a set of atoms (e.g. a reaction site): centre on their centroid, zoom so all fit with `padding` Å to spare, and "
             "rotate so the first two atoms lie in the screen plane on a horizontal line (first atom on the left) with the "
             "plane of all atoms facing the viewer. The slab covers the atoms' depth plus `padding`/2.",
             {"atoms": ("array", "list of atom specs (at least one); " + ATOM_SPEC), **MOL,
              "padding": ("number", "Å of context around the atoms (default 4)"),
              "slab": ("number", "depth half-thickness in Å (default: atoms' depth + 2 Å, at least 3)"),
              "orient": ("boolean", "rotate the view (default true)"),
              "roll": ("number", "extra roll about the viewing axis in degrees")}, view_only=True)
    def frame_atoms(self, atoms, mol=None, padding=4.0, slab=None, orient=True, roll=0.0):
        from scipy.spatial.transform import Rotation
        if isinstance(atoms, str):
            atoms = [atoms]
        pts = np.array([self.point(a, mol) for a in atoms])
        c = pts.mean(0)
        extent = float(np.linalg.norm(pts - c, axis=1).max()) if len(pts) > 1 else 0.0
        q = None
        cur = self.view.get("orient")
        view_z = Rotation.from_quat(cur).as_matrix()[2] if cur else np.array([0.0, 0.0, 1.0])
        z = view_z
        if orient and len(pts) >= 2 and np.linalg.norm(pts[1] - pts[0]) > 1e-3:
            x = (pts[1] - pts[0]) / np.linalg.norm(pts[1] - pts[0])
            z = None
            if len(pts) >= 3:
                _, sv, vt = np.linalg.svd(pts - c)
                if sv[1] > 0.2:                       # not collinear: look at the plane of the atoms
                    z = vt[2]
            if z is None:                             # keep the current viewing direction as far as possible
                z = view_z
            z = z - x * (z @ x)
            if np.linalg.norm(z) < 1e-3:
                z = np.cross(x, [0.0, 1.0, 0.0] if abs(x[1]) < 0.9 else [1.0, 0.0, 0.0])
            z = z / np.linalg.norm(z)
            if z @ view_z < 0:
                z = -z
            y = np.cross(z, x)
            r = Rotation.from_euler("z", float(roll or 0), degrees=True) * Rotation.from_matrix(np.array([x, y, z]))
            q = [float(v) for v in r.as_quat()]
        depth = float(np.abs((pts - c) @ z).max()) if len(pts) > 1 else 0.0
        slab = float(slab) if slab else max(3.0, depth + 2.0)
        fit = extent + float(padding)
        self.view["center"] = c.tolist()
        if q:
            self.view["orient"] = q
        self.broadcast({"type": "view_cmd", "center": c.tolist(), "fit_radius": fit, "slab": slab, **({"orient": q} if q else {})})
        out = {"center": [round(float(v), 3) for v in c], "fit_radius": round(fit, 2), "slab": round(slab, 2), "oriented": q is not None}
        if len(pts) >= 2:
            out["distance_first_pair"] = round(float(np.linalg.norm(pts[1] - pts[0])), 3)
        return out

    @command("Change a map's display: visibility, contour level (sigma), radius (Å), colour; or make it "
             "the map used for real-space refinement.",
             {"map_id": ("integer", "map id"), "visible": ("boolean", ""), "level_sigma": ("number", ""),
              "radius": ("number", "contour box half width"), "color": ("string", "css colour"),
              "refinement_map": ("boolean", "use for real-space refinement")}, view_only=True)
    def set_map(self, map_id, visible=None, level_sigma=None, radius=None, color=None, refinement_map=None):
        x = self.maps[int(map_id)]
        if visible is not None:
            x.visible = bool(visible)
        if level_sigma is not None:
            x.level_sigma = float(level_sigma)
        if radius is not None:
            x.radius = float(radius)
        if color:
            x.color = color
        if refinement_map:
            self.refinement_map = x.id
        self.push_maps()
        return x.info()

    @command("Rename a map (the name shown in the panel, state and results).",
             {"map_id": ("integer", ""), "name": ("string", "new name")}, view_only=True)
    def rename_map(self, map_id, name):
        x = self.maps[int(map_id)]
        x.name = str(name)
        self.push_maps()
        return x.info()

    @command("Close a map.", {"map_id": ("integer", "")})
    def close_map(self, map_id):
        self._require_no_rsr()
        map_id = int(map_id)
        if self.maps.pop(map_id, None) is None:
            raise ValueError(f"no map {map_id}")
        if self.refinement_map == map_id:
            self.refinement_map = None
        if self.diff_map == map_id:
            self.diff_map = None
        self.push_maps()
        self.push_rsr_settings()
        return "ok"

    @command("Close a molecule.", {**MOL})
    def close_molecule(self, mol=None):
        self._require_no_rsr()
        m = self.mol(mol)
        del self.molecules[m.id]
        self.active_mol = next(iter(self.molecules), None)
        self.broadcast({"type": "close_molecule", "mol": m.id})
        return "ok"

    @command("Set which molecule edits apply to.", {"mol": ("integer", "molecule id")}, view_only=True)
    def set_active_molecule(self, mol):
        self.mol(mol)
        self.active_mol = int(mol)
        self.broadcast({"type": "active_molecule", "mol": self.active_mol})
        return self.active_mol

    def push_display(self, m):
        self.broadcast({"type": "mol_display", "mol": m.id, **m.display()})

    @command("Draw only whole residues within `radius` Å of the given atoms (in every visible molecule, or in `mols`), "
             "like PyMOL's byres ... within: for figures and uncluttered views. clear=true draws everything again. Saved "
             "views and tour stops keep the zone.",
             {"atoms": ("array", "atom specs the zone is around"), "radius": ("number", "Å (default 5)"),
              "mols": ("array", "molecule ids (default: all visible)"), "clear": ("boolean", "draw all residues again")},
             view_only=True)
    def show_zone(self, atoms=None, radius=5.0, mols=None, clear=False):
        targets = [self.molecules[int(i)] for i in mols] if mols else [m for m in self.molecules.values() if m.visible]
        if clear:
            for m in targets if mols else self.molecules.values():
                m.zone = None
                self.push_display(m)
            return {"zone": None}
        if not atoms:
            raise ValueError("give atoms (or clear=true)")
        pts = np.array([self.point(a) for a in ([atoms] if isinstance(atoms, str) else atoms)])
        out = {}
        for m in targets:
            if not len(m.xyz):
                continue
            d = cKDTree(pts).query(m.xyz)[0]
            m.zone = sorted({k[:3] for k, dd in zip(m.keys, d) if dd <= float(radius)})
            self.push_display(m)
            out[m.name] = [f"{c}/{n}{i}" for c, n, i in m.zone]
        return {"radius": radius, "residues": out}

    @command("Show or hide a molecule, set its carbon colour, drawing style or hydrogens. The panel checkboxes call this too, so "
             "`state` always reports what the user sees.",
             {"mol": ("integer", "molecule id"), "visible": ("boolean", ""), "color": ("string", "css colour for carbons, "
              "e.g. '#ff80c0' or 'cyan'; 'default' resets"), "style": ("string", "'sticks', 'lines' or 'cartoon' (main chain as a tube, side chains as sticks)"),
              "hydrogens": ("boolean", "show hydrogens")}, view_only=True)
    def set_molecule(self, mol, visible=None, color=None, style=None, hydrogens=None):
        m = self.mol(mol)
        if visible is not None:
            m.visible = bool(visible)
        if color:
            m.color = None if color == "default" else str(color)
        if style:
            if style not in ("sticks", "lines", "cartoon"):
                raise ValueError("style must be 'sticks', 'lines' or 'cartoon' (main chain as a tube, side chains as sticks)")
            m.style = style
        if hydrogens is not None:
            m.hydrogens = bool(hydrogens)
        self.push_display(m)
        return {"mol": m.id, **m.display()}

    @command("Show only this molecule (hide all others).", {"mol": ("integer", "molecule id")}, view_only=True)
    def solo(self, mol):
        keep = self.mol(mol).id
        for m in self.molecules.values():
            m.visible = m.id == keep
            self.push_display(m)
        return {m.id: m.visible for m in self.molecules.values()}

    # ====================================================== map analysis
    @command("List strongest peaks (positive and negative) of a map, with the nearest atom of the active "
             "molecule. Useful for difference maps in time-resolved work.",
             {"map_id": ("integer", ""), "sigma": ("number", "threshold (default 3.5)"),
              "max_peaks": ("integer", "default 30"), "max_distance": ("number", "only peaks within this "
                                                                                 "distance of the model (default 4 Å; 0 = anywhere)"),
              **MOL},
             view_only=True)
    def peaks(self, map_id, sigma=3.5, max_peaks=30, max_distance=4.0, mol=None):
        """Each peak carries its rank among all peaks of the same sign in the whole cell ('rank' of 'of'), so a peak can be
        judged against the noise ("the 15th of 560 peaks above 3.5σ")."""
        from .maps import SymMapper

        x = self.maps[int(map_id)]
        m = self.mol(mol)
        sym = SymMapper(m.st, m.xyz, mask=np.array([e != "H" for e in m.elements]))
        out, totals = [], {}
        for neg in ([False, True] if x.kind == "diff" else [False]):
            found = x.peaks(sigma, negative=neg)
            totals["negative" if neg else "positive"] = len(found)
            n_sign = 0
            for rank, (xyz, h) in enumerate(found, 1):
                pos, d, ai = sym.nearest(xyz)
                if (max_distance and d > max_distance) or any(np.linalg.norm(np.array(o["xyz"]) - pos) < 0.7 for o in out):
                    continue
                out.append({"height_sigma": round(-h if neg else h, 2), "xyz": [round(float(v), 2) for v in pos],
                            "nearest": self._atom_label(m, ai), "distance": round(d, 2),
                            "rank": rank, "of": len(found)})
                n_sign += 1
                if n_sign >= 2 * max_peaks:
                    break
        out.sort(key=lambda p: -abs(p["height_sigma"]))
        self.say(f"peaks in [{x.id}] {x.name} above {sigma}σ, whole cell: " + ", ".join(f"{n} {k}" for k, n in totals.items()))
        return out[:max_peaks]

    @command("List ligands (non-polymer, non-water residues) with a guess whether each is relevant: common "
             "additives (PEG, EDO, GOL, SO4, buffers, Na/Cl, detergents, lipids/alkanes) and uncoordinated metals are flagged not "
             "relevant.", {**MOL, "relevant_only": ("boolean", "only the relevant ones")}, view_only=True)
    def ligands(self, mol=None, relevant_only=False):
        out = edit.list_ligands(self.mol(mol))
        return [l for l in out if l["relevant"]] if relevant_only else out

    @command("Correlation between two maps around residues (e.g. compare an Fo-Fo map with an Fc-Fc map). "
             "Computed on a grid within radius of the residues' atoms.",
             {"map_a": ("integer", ""), "map_b": ("integer", ""), **RANGE,
              "radius": ("number", "Å around atoms (default 2.5)")}, view_only=True)
    def map_correlation(self, map_a, map_b, chain, start, end, radius=2.5):
        m = self.mol()
        idx = [i for i, k in enumerate(m.keys) if k[0] == chain and start <= k[1] <= end]
        if not idx:
            raise ValueError("no atoms in range")
        pts = m.xyz[idx]
        a, b = self.maps[int(map_a)], self.maps[int(map_b)]
        lo, hi = pts.min(0) - radius, pts.max(0) + radius
        g = np.mgrid[lo[0]:hi[0]:0.4, lo[1]:hi[1]:0.4, lo[2]:hi[2]:0.4].reshape(3, -1).T
        d, _ = cKDTree(pts).query(g)
        g = g[d < radius]
        va = np.array([a.value_at(p) for p in g])
        vb = np.array([b.value_at(p) for p in g])
        return {"cc": float(np.corrcoef(va, vb)[0, 1]), "n_points": int(len(g)),
                "rms_a_sigma": float(va.std() / a.rms), "rms_b_sigma": float(vb.std() / b.rms)}

    @command("Map value (in sigma of each map's whole-cell rms) at each atom of a residue range, for the given maps (default: "
             "the visible maps). Visibility is not changed. Give `atoms` instead of a range to sample a list of atom specs.",
             {"chain": RANGE["chain"], "start": RANGE["start"], "end": RANGE["end"], **MAPS, **MOL,
              "atoms": ("array", "list of atom specs instead of a residue range; " + ATOM_SPEC)}, view_only=True)
    def density_at_atoms(self, chain=None, start=None, end=None, maps=None, mol=None, atoms=None):
        m = self.mol(mol)
        if atoms:
            pairs = []
            for a in ([atoms] if isinstance(atoms, (str, dict)) else atoms):
                mm, idx = self.atoms(a, mol)
                pairs += [(mm, i) for i in idx]
        else:
            if chain is None or start is None:
                raise ValueError("give chain, start (and end), or atoms")
            end = start if end is None else end
            pairs = [(m, i) for i, k in enumerate(m.keys) if k[0] == chain and int(start) <= k[1] <= int(end)]
            if not pairs:
                raise ValueError(f"no atoms in {chain}/{start}-{end} of molecule {m.id}")
        out = {}
        for x in self.map_list(maps):
            out[self.map_key(x)] = {(self._atom_label(mm, i) if mm is m else f"{mm.id}#{self._atom_label(mm, i)}"):
                                    self.sigma(x, mm.xyz[i]) for mm, i in pairs}
        return out

    @command("Distance between two atoms, possibly in different molecules. Without altlocs, every pair of conformers is listed.",
             {"a": ("string", ATOM_SPEC), "b": ("string", ATOM_SPEC), **MOL}, view_only=True)
    def distance(self, a, b, mol=None):
        ma, ia = self.atoms(a, mol)
        mb, ib = self.atoms(b, mol)
        lab = lambda mm, i: f"{mm.id}#{self._atom_label(mm, i)}"
        pairs = [{"a": lab(ma, i), "b": lab(mb, j), "distance": round(float(np.linalg.norm(ma.xyz[i] - mb.xyz[j])), 3)}
                 for i in ia for j in ib if not (ma is mb and i == j)]
        if not pairs:
            raise ValueError("no atom pairs")
        if len(pairs) == 1:
            return pairs[0]
        return {"pairs": pairs, "min": min(p["distance"] for p in pairs), "max": max(p["distance"] for p in pairs)}

    @command("Density profile along the line between two atoms (or points): map values in σ at n evenly spaced points, the "
             "lowest value between the ends and its ratio to the mean of the two end values (≈1: continuous density, a bond; "
             "≪1: a gap). Uses the first conformer unless an altloc is given.",
             {"a": ("string", ATOM_SPEC), "b": ("string", ATOM_SPEC), **MAPS, **MOL,
              "n": ("integer", "number of points including the ends (default 11)")}, view_only=True)
    def density_profile(self, a, b, maps=None, mol=None, n=11):
        pa, pb = self.point(a, mol), self.point(b, mol)
        n = max(3, int(n))
        ts = np.linspace(0, 1, n)
        pts = pa[None] + ts[:, None] * (pb - pa)[None]
        out = {"a": str(a), "b": str(b), "distance": round(float(np.linalg.norm(pb - pa)), 3),
               "t": [round(float(t), 3) for t in ts], "maps": {}}
        for x in self.map_list(maps):
            v = [self.sigma(x, p) for p in pts]
            inner = v[1:-1]
            k = int(np.argmin(inner))
            ends = (v[0] + v[-1]) / 2
            out["maps"][self.map_key(x)] = {
                "values": v, "min_between": inner[k], "min_at_t": round(float(ts[k + 1]), 3), "ends_mean": round(ends, 2),
                "min_over_ends": round(inner[k] / ends, 2) if abs(ends) > 1e-6 else None}
        return out

    @command("Map values (σ) at a point: x, y, z, or an atom spec plus an optional offset [dx, dy, dz] in Å.",
             {"x": ("number", ""), "y": ("number", ""), "z": ("number", ""), "atom": ("string", ATOM_SPEC),
              "offset": ("array", "[dx, dy, dz] in Å added to the atom position"), **MAPS, **MOL}, view_only=True)
    def density_at_point(self, x=None, y=None, z=None, atom=None, offset=None, maps=None, mol=None):
        if atom is not None:
            p = self.point(atom, mol)
        elif None not in (x, y, z):
            p = np.array([float(x), float(y), float(z)])
        else:
            raise ValueError("give x, y, z or atom")
        if offset:
            p = p + np.asarray([float(v) for v in offset])
        return {"xyz": [round(float(v), 3) for v in p], "values": {self.map_key(xm): self.sigma(xm, p) for xm in self.map_list(maps)}}

    # ================================================= views, tours, labels
    def _fresh_view(self):
        """Ask the browser for its current view (messages are ordered, so a preceding goto is already applied)."""
        try:
            v = self.ask_client("view_request", {}, timeout=3)
            if isinstance(v, dict):
                self.view.update({k: v[k] for k in ("center", "orient", "zoom", "slab") if k in v})
        except Exception:
            pass          # no browser connected: use the last reported view
        return self.view

    def _real_display(self, m):
        """Display state without the figure preview's look."""
        if self.preview is not None and m.id in self.preview["saved"]:
            st, zn = self.preview["saved"][m.id]
            return m.display(style=st, zone=zn)
        return m.display()

    def _annotations_with_clicks(self):
        """API labels plus the labels the user made by clicking atoms in the viewer (so views and tour stops keep them)."""
        ann = json.loads(json.dumps(self.annotations))
        for mid, i in self.view.get("labelled") or []:
            m = self.molecules.get(int(mid))
            if m is not None and 0 <= int(i) < len(m.keys):
                spec = f"{m.id}#{self._atom_label(m, int(i))}"
                if spec not in ann["labels"]:
                    ann["labels"].append(spec)
        return ann

    def _snapshot(self):
        v = self._fresh_view()
        return {
            "center": v.get("center"), "orient": v.get("orient"), "zoom": v.get("zoom"), "slab": v.get("slab"),
            "active_mol": {"id": self.active_mol, "name": self.molecules[self.active_mol].name} if self.active_mol in self.molecules else None,
            "molecules": [{"id": m.id, "name": m.name, **self._real_display(m)} for m in self.molecules.values()],
            "maps": [{"id": x.id, "name": x.name, "visible": x.visible, "level_sigma": x.level_sigma, "radius": x.radius,
                      "color": x.color} for x in self.maps.values()],
            "annotations": self._annotations_with_clicks(),
            "slice": json.loads(json.dumps(self.slice)),
        }

    @staticmethod
    def _match(items, d, used=()):
        """Object with the saved id if it still has the saved name, else the first unused one with that name (tours in a new
        session)."""
        x = items.get(d.get("id"))
        if x is not None and x.name == d.get("name") and x.id not in used:
            return x
        return next((y for y in items.values() if y.name == d.get("name") and y.id not in used), None)

    def _apply_snapshot(self, snap):
        used = set()
        for d in snap.get("molecules", []):
            m = self._match(self.molecules, d, used)
            if m:
                used.add(m.id)
                m.visible, m.color, m.style, m.hydrogens = d.get("visible", True), d.get("color"), d.get("style"), d.get("hydrogens")
                zr = d.get("zone_residues")
                m.zone = None if zr is None else [tuple(k) for k in zr]
                self.push_display(m)
        named = set()
        for d in snap.get("maps", []):
            x = self._match(self.maps, d, named)
            if x:
                named.add(x.id)
                x.visible, x.level_sigma = d.get("visible", x.visible), d.get("level_sigma", x.level_sigma)
                x.radius, x.color = d.get("radius", x.radius), d.get("color") or x.color
        for x in self.maps.values():          # maps not in the snapshot are hidden
            if x.id not in named:
                x.visible = False
        self.push_maps()
        act = snap.get("active_mol")
        if isinstance(act, dict) and self._match(self.molecules, act) is not None:
            self.set_active_molecule(self._match(self.molecules, act).id)
        ann = snap.get("annotations")
        if ann is not None:
            self.clear_annotations()
            for a in ann.get("labels", []):
                try:
                    self.label_atoms([a])
                except ValueError:
                    pass
            for e in ann.get("distances", []):
                try:
                    self.show_distance(e[0], e[1], style=e[2] if len(e) > 2 else "distance",
                                       label=e[3] if len(e) > 3 else None)
                except ValueError:
                    pass
        if isinstance(snap.get("slice"), dict):
            self._set_slice(snap["slice"])
        msg = {"type": "view_cmd"}
        for k in ("center", "orient", "zoom", "slab"):
            if snap.get(k) is not None:
                msg[k] = snap[k]
        self.view.update({k: msg[k] for k in ("center", "orient", "zoom", "slab") if k in msg})
        self.broadcast(msg)

    # ------------------------------------------------------------- map slices
    def _set_slice(self, d):
        sl = self.slice
        if d.get("style") in ("mesh", "slice"):
            sl["style"] = d["style"]
        if "map" in d:
            sl["map"] = None if d["map"] is None else int(d["map"])
        if "lock" in d:
            lk = d["lock"]
            sl["lock"] = None if not lk else {"center": [float(v) for v in lk["center"]],
                                              "orient": [float(v) for v in lk["orient"]]}
        for k in ("depth", "thickness"):
            if d.get(k) is not None:
                sl[k] = float(d[k])
        self.broadcast({"type": "slice", **sl})

    def slice_from_client(self, m):
        """The user changed the slice in the viewer."""
        self._set_slice({k: m[k] for k in ("style", "map", "lock", "depth", "thickness") if k in m})

    def _slice_map(self):
        sl = self.slice
        if sl["map"] in self.maps:
            return self.maps[sl["map"]]
        return next((x for x in self.maps.values() if x.visible), None)

    def _slice_plane(self):
        """(centre on the plane, axes rows x, y, normal) of the slice as drawn."""
        from scipy.spatial.transform import Rotation
        sl = self.slice
        if sl["lock"]:
            c, q = np.array(sl["lock"]["center"]), sl["lock"]["orient"]
        else:
            v = self.view
            c, q = np.array(v.get("center") or [0, 0, 0], float), v.get("orient")
        axes = Rotation.from_quat(q).as_matrix() if q else np.eye(3)
        return c + sl["depth"] * axes[2], axes

    def _slice_stats(self, x):
        """Strongest positive and negative value on the slice (with thickness: within it), with the nearest atoms."""
        c, axes = self._slice_plane()
        r, sl = x.radius, self.slice
        step = 0.25
        n = int(2 * r / step) + 1
        nt = 13 if sl["thickness"] > 0.01 else 1
        dt = sl["thickness"] / 12 if nt > 1 else 0.0
        arr = np.zeros((n, n, nt), dtype=np.float32)
        tr = gemmi.Transform()
        tr.mat.fromlist(np.column_stack([axes[0] * step, axes[1] * step, axes[2] * dt]).tolist())
        tr.vec.fromlist((c - r * axes[0] - r * axes[1] - (sl["thickness"] / 2 if nt > 1 else 0) * axes[2]).tolist())
        x.grid.interpolate_values(arr, tr)
        sig = (arr - x.mean) / x.rms
        mols = [m for m in self.molecules.values() if m.visible and len(m.xyz)]
        out = {}
        for label, idx in (("max", np.unravel_index(np.argmax(sig), sig.shape)),
                           ("min", np.unravel_index(np.argmin(sig), sig.shape))):
            i, j, k = (int(v) for v in idx)
            p = tr.apply(gemmi.Vec3(i, j, k))
            p = np.array([p.x, p.y, p.z])
            e = {"sigma": round(float(sig[i, j, k]), 2), "xyz": [round(float(v), 2) for v in p]}
            best = None
            for m in mols:
                d, ai = cKDTree(m.xyz).query(p)
                if best is None or d < best[0]:
                    best = (d, m, ai)
            if best:
                e["nearest"] = (f"{best[1].id}#" if len(mols) > 1 else "") + self._atom_label(best[1], best[2])
                e["distance"] = round(float(best[0]), 2)
            out[label] = e
        return {"map": self.map_key(x), "center": [round(float(v), 3) for v in c],
                "normal": [round(float(v), 4) for v in axes[2]], "half_width": r, **out}

    @command("Map slices, as in shared pages: style='slice' draws the map as a plane (green +, red -; blue for density "
             "maps; brighter towards 3x level, contour lines at 1, 2, 4... x level) instead of mesh contours. The plane faces "
             "the viewer through the centre and follows the view, unless it is fixed in the molecule (fix=true, or atoms=...). "
             "Returns where the plane is and the strongest positive / negative value on it, with the nearest atoms.",
             {"style": ("string", "'slice' or 'mesh'"),
              "map_id": ("integer", "map drawn on the slice (default: the first visible map; 0 = back to that default)"),
              "atoms": ("array", "plane through these atoms (3 or more: best plane; 2: the plane holding that line that "
                                 "faces the viewer most); turns the view onto it and fixes it; implies style='slice'"),
              "fix": ("boolean", "true: fix the plane where it is now; false: let it follow the view again"),
              "depth": ("number", "move the plane along its normal (Å)"),
              "thickness": ("number", "show the strongest value within ± thickness/2 of the plane (Å; 0 = the plane)"),
              **MOL}, view_only=True)
    def slice(self, style=None, map_id=None, atoms=None, fix=None, depth=None, thickness=None, mol=None):
        d = {}
        if style is not None:
            if style not in ("slice", "mesh"):
                raise ValueError("style: 'slice' or 'mesh'")
            d["style"] = style
        if map_id is not None:
            if int(map_id) and int(map_id) not in self.maps:
                raise ValueError(f"no map {map_id}; maps: {list(self.maps)}")
            d["map"] = int(map_id) or None
        if atoms:
            self.frame_atoms(atoms, mol)
            d["lock"] = {"center": list(self.view["center"]), "orient": self.view.get("orient") or [0, 0, 0, 1]}
            d.setdefault("style", "slice")
            d["depth"] = 0.0
        elif fix is True:
            v = self._fresh_view()
            d["lock"] = {"center": list(v["center"]), "orient": v.get("orient") or [0, 0, 0, 1]}
        elif fix is False:
            d["lock"] = None
        if depth is not None:
            d["depth"] = depth
        if thickness is not None:
            d["thickness"] = max(0.0, float(thickness))
        if not atoms:
            self._fresh_view()
        self._set_slice(d)
        out = {"slice": self.slice}
        x = self._slice_map()
        if x is None:
            out["note"] = "no map is visible: the plane is drawn empty"
        elif self.slice["style"] == "slice":
            out["plane"] = self._slice_stats(x)
        return out

    @command("Save the current view under a name: centre, rotation, zoom, slab, visible molecules (and colours), visible maps "
             "and their levels, labels/distances set through the API. Stored in <workdir>/.moorlight/views.json.",
             {"name": ("string", "view name")}, view_only=True)
    def save_view(self, name):
        self.views[str(name)] = self._snapshot()
        self._write_json("views.json", self.views)
        return {"saved": name, "views": list(self.views)}

    @command("Restore a saved view (see save_view). Maps and molecules are matched by name.",
             {"name": ("string", "view name")}, view_only=True)
    def restore_view(self, name):
        if name not in self.views:
            raise ValueError(f"no view {name!r}; saved views: {list(self.views)}")
        self._apply_snapshot(self.views[name])
        return {"restored": name}

    @command("List saved views.", view_only=True)
    def list_views(self):
        return list(self.views)

    def push_tour(self):
        self.broadcast({"type": "tour", "title": self.tour["title"], "index": self.tour_index,
                        "stops": [{"name": st.get("name", ""), "text": st.get("text", "")} for st in self.tour["stops"]]})

    @command("Add the current view as a stop of the tour, with a short text shown to the user next to the view. A tour can be "
             "saved (tour_save) and stepped through in the viewer with ◀ ▶ without an agent.",
             {"text": ("string", "what to look at, one to three sentences"), "name": ("string", "short stop name"),
              "index": ("integer", "insert at this position (default: append)"),
              "replace": ("boolean", "replace the stop at `index` instead of inserting"),
              "title": ("string", "tour title (optional)")}, view_only=True)
    def tour_add(self, text="", name=None, index=None, replace=False, title=None):
        stop = {"name": name or f"stop {len(self.tour['stops']) + 1}", "text": text, "view": self._snapshot()}
        stops = self.tour["stops"]
        if title:
            self.tour["title"] = title
        if index is None:
            stops.append(stop)
            self.tour_index = len(stops) - 1
        else:
            index = int(index)
            if replace:
                stops[index] = stop
            else:
                stops.insert(index, stop)
            self.tour_index = index
        self.push_tour()
        return {"stops": len(stops), "index": self.tour_index}

    @command("Go to a tour stop: index (0-based) or step (+1 next, -1 previous).",
             {"index": ("integer", "stop index"), "step": ("integer", "+1 / -1")}, view_only=True)
    def tour_goto(self, index=None, step=None):
        stops = self.tour["stops"]
        if not stops:
            raise ValueError("no tour loaded")
        i = int(index) if index is not None else self.tour_index + int(step or 1)
        i = max(0, min(len(stops) - 1, i))
        self.tour_index = i
        self._apply_snapshot(stops[i]["view"])
        self.push_tour()
        return {"index": i, "of": len(stops), "name": stops[i].get("name"), "text": stops[i].get("text")}

    @command("Save the tour as JSON (stops with views and texts, plus the files needed to show them).",
             {"path": ("string", "output .json path"), "title": ("string", "tour title")}, view_only=True)
    def tour_save(self, path, title=None):
        if title:
            self.tour["title"] = title
        files = {"models": [{"path": m.path, "name": m.name} for m in self.molecules.values() if m.path],
                 "maps": [{"path": getattr(x, "path", None), "name": x.name, "f": x.f, "phi": x.phi, "kind": x.kind,
                           "d_min": x.d_min} for x in self.maps.values() if getattr(x, "path", None)]}
        p = self._path(path)
        with open(p, "w") as fh:
            json.dump({**self.tour, "files": files}, fh, indent=1)
        return p

    @command("Load a tour JSON and go to its first stop. Missing models and maps listed in the file are loaded first.",
             {"path": ("string", "tour .json"), "load_files": ("boolean", "load missing models/maps (default true)")},
             view_only=True)
    def tour_load(self, path, load_files=True):
        with open(self._path(path)) as fh:
            t = json.load(fh)
        if load_files:
            names = {m.name for m in self.molecules.values()}
            for f in t.get("files", {}).get("models", []):
                if f["name"] not in names and f.get("path") and os.path.exists(f["path"]):
                    self.load_model(f["path"])
            mnames = {x.name for x in self.maps.values()}
            for f in t.get("files", {}).get("maps", []):
                if f["name"] in mnames or not f.get("path") or not os.path.exists(f["path"]):
                    continue
                if f.get("f"):
                    self.load_mtz(f["path"], columns=[f"{f['f']},{f['phi']},{f['kind']},{f['name']}"], d_min=f.get("d_min"))
                    if f.get("d_min"):   # load_mtz appended the d_min to the name; keep the saved one
                        list(self.maps.values())[-1].name = f["name"]
                else:
                    self.load_map(f["path"], difference=f.get("kind") == "diff", label=f["name"])
        self.tour = {"title": t.get("title", ""), "stops": t.get("stops", [])}
        self.tour_index = -1
        if self.tour["stops"]:
            return self.tour_goto(0)
        self.push_tour()
        return {"stops": 0}

    @command("Remove all tour stops.", view_only=True)
    def tour_clear(self):
        self.tour = {"title": "", "stops": []}
        self.tour_index = -1
        self.push_tour()
        return "ok"

    def _ref(self, spec, mol=None):
        m, idx = self.atoms(spec, mol, all_altlocs=False)
        if len({m.keys[i][3] for i in idx}) > 1:
            raise ValueError(f"{spec!r} names a residue; give an atom")
        return m, idx[0]

    @command("Label atoms in the 3D view (labels also appear in screenshots).",
             {"atoms": ("array", "list of atom specs; " + ATOM_SPEC), **MOL,
              "clear": ("boolean", "remove existing labels first")}, view_only=True)
    def label_atoms(self, atoms, mol=None, clear=False):
        if isinstance(atoms, (str, dict)):
            atoms = [atoms]
        refs = [self._ref(a, mol) for a in atoms]
        if clear:
            self.annotations["labels"] = []
        self.annotations["labels"] += [f"{m.id}#{self._atom_label(m, i)}" for m, i in refs]
        self.broadcast({"type": "annotate", "clear_labels": bool(clear), "labels": [{"mol": m.id, "idx": i} for m, i in refs]})
        return [self._atom_label(m, i) for m, i in refs]

    @command("Draw a dashed line between two atoms (also in screenshots and figures): style 'distance' (with its value) or "
             "'hbond' (thinner, blue, no value unless label=true).",
             {"a": ("string", ATOM_SPEC), "b": ("string", ATOM_SPEC),
              "style": ("string", "'distance' (default) or 'hbond'"),
              "label": ("boolean", "show the value (default: true for distance, false for hbond)"), **MOL}, view_only=True)
    def show_distance(self, a, b, mol=None, style="distance", label=None):
        if style not in ("distance", "hbond"):
            raise ValueError("style: 'distance' or 'hbond'")
        label = (style == "distance") if label is None else bool(label)
        (ma, ia), (mb, ib) = self._ref(a, mol), self._ref(b, mol)
        self.annotations["distances"].append([f"{ma.id}#{self._atom_label(ma, ia)}", f"{mb.id}#{self._atom_label(mb, ib)}",
                                              style, label])
        self.broadcast({"type": "annotate", "distances": [{"a": {"mol": ma.id, "idx": ia}, "b": {"mol": mb.id, "idx": ib},
                                                            "style": style, "label": label}]})
        return round(float(np.linalg.norm(ma.xyz[ia] - mb.xyz[ib])), 3)

    @command("Find and draw hydrogen bonds of the given residues or atoms: N/O pairs within max_dist Å to other residues "
             "(of the same molecule, waters included). Drawn as 'hbond' dashes; returns the list with distances.",
             {"atoms": ("array", "residue or atom specs, e.g. ['A/85', 'A/401']"),
              "max_dist": ("number", "Å (default 3.3)"), "label": ("boolean", "show the distances (default false)"),
              **MOL}, view_only=True)
    def show_hbonds(self, atoms, max_dist=3.3, label=False, mol=None):
        if isinstance(atoms, str):
            atoms = [atoms]
        found, seen = [], set()
        for spec in atoms:
            m, idx = self.atoms(spec, mol)
            polar = [i for i in idx if m.elements[i] in ("N", "O")]
            own = {m.keys[i][:3] for i in idx}
            tree = cKDTree(m.xyz)
            for i in polar:
                for j in tree.query_ball_point(m.xyz[i], float(max_dist)):
                    if m.elements[j] not in ("N", "O") or m.keys[j][:3] in own:
                        continue
                    ai, aj = m.keys[i][4], m.keys[j][4]
                    if ai and aj and ai != aj:                 # different conformers do not bond
                        continue
                    key = (m.id, min(i, j), max(i, j))
                    if key in seen:
                        continue
                    seen.add(key)
                    d = float(np.linalg.norm(m.xyz[i] - m.xyz[j]))
                    self.show_distance(f"{m.id}#{self._atom_label(m, i)}", f"{m.id}#{self._atom_label(m, j)}",
                                       style="hbond", label=label)
                    found.append({"a": self._atom_label(m, i), "b": self._atom_label(m, j), "distance": round(d, 2)})
        return sorted(found, key=lambda x: x["distance"])

    @command("Remove all labels and distance lines.", view_only=True)
    def clear_annotations(self):
        self.annotations = {"labels": [], "distances": []}
        self.view["labelled"] = []
        self.broadcast({"type": "annotate", "clear_labels": True, "clear_distances": True})
        return "ok"

    # ========================================================= refinement
    def _start_rsr(self, m, idx, altloc=None, weight=None):
        from .rsr import RSRJob
        use_map = self.rsr_target in ("map", "both")
        use_diff = self.rsr_target in ("difference", "both")
        xmap = self.maps.get(self.refinement_map) if (use_map and self.refinement_map) else None
        dmap = self.maps.get(self.diff_map) if (use_diff and self.diff_map) else None
        if use_diff and dmap is None:
            raise ValueError("difference target needs a difference map (set one with set_refinement_options)")
        if xmap is None and dmap is None:
            self.say("warning: no refinement map set – regularising geometry only")
        dark = self.molecules.get(self.dark_model) if self.dark_model not in (None, m.id) else None
        last = [0.0]

        def on_update(job):
            now = time.time()
            if now - last[0] > 0.04:
                last[0] = now
                self.broadcast({"type": "rsr_coords", "mol": m.id, "idx": job.moving_mol.tolist(),
                                "xyz": np.round(job.moving_coords(), 3).ravel().tolist()})

        job = RSRJob(m, xmap, idx, weight=weight or self.rsr_weight, altloc=altloc,
                     cif_paths=self.cif_paths, on_update=on_update, nonbonded=self.rsr_nonbonded,
                     diff_map=dmap, diff_weight=self.rsr_diff_weight, dark_mol=dark)
        self.rsr = job
        self.broadcast({"type": "rsr_start", "mol": m.id, "idx": job.moving_mol.tolist(), "altloc": job.altloc})
        return job

    def _rsr_loop(self, job):
        while self.rsr is job:
            job.run(max_steps=400)
            st = job.status
            self.broadcast({"type": "rsr_coords", "mol": job.mol.id, "idx": job.moving_mol.tolist(),
                            "xyz": np.round(job.moving_coords(), 3).ravel().tolist()})
            self.broadcast({"type": "rsr_status", **{k: v for k, v in st.items() if k != "terms"}})
            # idle until a pull arrives or the job is finished
            while self.rsr is job and not job.pulls:
                time.sleep(0.03)

    @command("Real-space refine a residue range (or a sphere around one residue) against the refinement map "
             "using TorchRef geometry restraints. With accept=false (default) the refinement stays live so the "
             "user can pull atoms; call accept_refinement / reject_refinement afterwards.",
             {**RANGE, "altloc": ("string", "refine only this alternative conformer"),
              "sphere_radius": ("number", "instead of a range: refine all residues within this radius of 'start'"),
              "weight": ("number", "map weight (default 60)"),
              "accept": ("boolean", "accept automatically when converged"), **MOL})
    def refine(self, chain, start, end=None, altloc=None, sphere_radius=None, weight=None, accept=False, mol=None):
        self._require_no_rsr()
        m = self.mol(mol)
        end = start if end is None else end
        if sphere_radius:
            ch, ri, res = m.find_residue(chain, int(start))
            keys = m.neighbours_of_residues({(chain, int(start), "")}, float(sphere_radius))
            idx = [i for i, k in enumerate(m.keys) if k[:3] in keys and m.info[i][3] not in ("HOH",)]
        else:
            idx = [i for i, k in enumerate(m.keys) if k[0] == chain and int(start) <= k[1] <= int(end)]
        if not idx:
            raise ValueError("nothing to refine")
        job = self._start_rsr(m, idx, altloc, weight)
        if accept:
            st = job.run()
            self.rsr = None
            m.checkpoint()
            job.apply()
            self.push_model(m)
            self.broadcast({"type": "rsr_end"})
            self.say(f"refined {chain}/{start}-{end}: bond rmsd {st.get('bond_rmsd', 0):.3f} Å, "
                     f"density {st.get('mean_density_sigma', 0):.2f}σ, max shift {st.get('max_shift', 0):.2f} Å")
            return st
        self.rsr_thread = threading.Thread(target=self._rsr_loop, args=(job,), daemon=True)
        self.rsr_thread.start()
        self.say(f"refining {len(job.moving_mol)} atoms ({chain}/{start}-{end}{' alt ' + job.altloc if job.altloc else ''})"
                 f" – drag atoms to pull, Enter accepts, Esc rejects")
        return {"moving_atoms": len(job.moving_mol), "restraint_build_s": round(job.build_time, 2)}

    def refine_atoms(self, idx, altloc=None):
        """UI entry: refine the residues containing these atom indices."""
        self._require_no_rsr()
        m = self.mol()
        res = {m.keys[i][:3] for i in idx}
        all_idx = [i for i, k in enumerate(m.keys) if k[:3] in res]
        job = self._start_rsr(m, all_idx, altloc)
        self.rsr_thread = threading.Thread(target=self._rsr_loop, args=(job,), daemon=True)
        self.rsr_thread.start()
        return {"moving_atoms": len(job.moving_mol)}

    def pull(self, atom, xyz):
        job = self.rsr
        if job is not None:
            job.set_pull(int(atom), xyz)

    @command("Accept the running real-space refinement.")
    def accept_refinement(self):
        job = self.rsr
        if job is None:
            return "nothing to accept"
        job.running = False
        self.rsr = None
        if self.rsr_thread:
            self.rsr_thread.join(5)
        job.pulls.clear()
        job.run(max_steps=5)  # settle after the last pull
        st = job.geometry_stats()
        job.mol.checkpoint()
        job.apply()
        self.push_model(job.mol)
        self.broadcast({"type": "rsr_end"})
        msg = f"accepted: bond rmsd {st.get('bond_rmsd', 0):.3f} Å"
        if "mean_density_sigma" in st:
            msg += f", density {st['mean_density_sigma']:.2f}σ"
        if "diff_cc" in st:
            msg += f", difference CC {st['diff_cc']:.2f}"
        self.say(msg)
        return st

    @command("Reject the running real-space refinement.")
    def reject_refinement(self):
        job = self.rsr
        if job is None:
            return "nothing to reject"
        job.running = False
        self.rsr = None
        self.broadcast({"type": "rsr_end"})
        self.push_model(job.mol)
        self.say("refinement rejected")
        return "rejected"

    def rsr_settings(self):
        return {"target": self.rsr_target, "weight": self.rsr_weight, "diff_weight": self.rsr_diff_weight,
                "nonbonded": self.rsr_nonbonded, "refinement_map": self.refinement_map,
                "diff_map": self.diff_map, "dark_model": self.dark_model}

    def push_rsr_settings(self):
        self.broadcast({"type": "rsr_settings", "settings": self.rsr_settings()})

    @command("Real-space refinement settings. target: 'map' (fit atoms into the refinement map), "
             "'difference' (fit the model's light-minus-dark difference density to a difference map such as "
             "WDF; dark reference = dark_model, else other altlocs of the residue, else starting coordinates) "
             "or 'both'. nonbonded: atoms repel each other (default off). dark_model 0 clears it.",
             {"target": ("string", "map | difference | both"), "weight": ("number", "map weight (60)"),
              "diff_weight": ("number", "difference weight (60)"), "nonbonded": ("boolean", "atoms repel"),
              "refinement_map": ("integer", "map id for the map target"),
              "diff_map": ("integer", "map id for the difference target"),
              "dark_model": ("integer", "molecule id of the dark state (0 = none)")})
    def set_refinement_options(self, target=None, weight=None, diff_weight=None, nonbonded=None,
                               refinement_map=None, diff_map=None, dark_model=None):
        if target is not None:
            if target not in ("map", "difference", "both"):
                raise ValueError("target must be map, difference or both")
            self.rsr_target = target
        if weight is not None:
            self.rsr_weight = float(weight)
        if diff_weight is not None:
            self.rsr_diff_weight = float(diff_weight)
        if nonbonded is not None:
            self.rsr_nonbonded = bool(nonbonded)
        if refinement_map is not None:
            self.refinement_map = int(refinement_map) or None
            self.push_maps()
        if diff_map is not None:
            self.diff_map = int(diff_map) or None
        if dark_model is not None:
            self.dark_model = int(dark_model) or None
        self.push_rsr_settings()
        return self.rsr_settings()

    @command("Set the default map weight for real-space refinement (Coot-like, default 60).",
             {"weight": ("number", "")})
    def set_refinement_weight(self, weight):
        self.rsr_weight = float(weight)
        return self.rsr_weight

    @command("Fix atoms so real-space refinement does not move them.",
             {**RANGE, "atom": ("string", "only this atom name"), **MOL})
    def fix_atoms(self, chain, start, end=None, atom=None, mol=None):
        m = self.mol(mol)
        end = start if end is None else end
        n = 0
        for k in m.keys:
            if k[0] == chain and start <= k[1] <= end and (atom is None or k[3] == atom):
                m.fixed.add(k)
                n += 1
        self.push_model(m)
        return f"fixed {n} atoms"

    def fix_indices(self, idx, fix=True):
        m = self.mol()
        for i in idx:
            (m.fixed.add if fix else m.fixed.discard)(m.keys[i])
        self.push_model(m)

    @command("Release fixed atoms (all if no range given).", {"chain": RANGE["chain"], "start": RANGE["start"],
                                                               "end": RANGE["end"], **MOL})
    def unfix_atoms(self, chain=None, start=None, end=None, mol=None):
        m = self.mol(mol)
        if chain is None:
            m.fixed.clear()
        else:
            end = start if end is None else end
            m.fixed = {k for k in m.fixed if not (k[0] == chain and start <= k[1] <= end)}
        self.push_model(m)
        return f"{len(m.fixed)} atoms still fixed"

    # ============================================================ building
    @command("Add an alternative conformer to a residue (side chain + CA, or whole residue). Returns the new "
             "altloc; then refine with altloc=<new> to move it.",
             {**RES, "whole_residue": ("boolean", ""), "occupancy": ("number", "occupancy of new conformer (0.5)"),
              **MOL})
    def add_altconf(self, chain, resseq, whole_residue=False, occupancy=0.5, mol=None):
        self._require_no_rsr()
        m = self.mol(mol)
        m.checkpoint()
        new = edit.add_altconf(m, chain, int(resseq), whole_residue=whole_residue, occ_new=float(occupancy))
        self._edited(m, f"{chain}/{resseq}: added conformer {new}")
        return new

    @command("Delete one alternative conformer of a residue.", {**RES, "altloc": ("string", ""), **MOL})
    def remove_altconf(self, chain, resseq, altloc, mol=None):
        self._require_no_rsr()
        m = self.mol(mol)
        m.checkpoint()
        edit.remove_altconf(m, chain, int(resseq), altloc)
        return self._edited(m, f"{chain}/{resseq}: removed conformer {altloc}")

    @command("List library rotamers of a residue, ranked by fit to the refinement map.",
             {**RES, "altloc": ("string", ""), **MOL}, view_only=True)
    def rotamers(self, chain, resseq, altloc="", mol=None):
        m = self.mol(mol)
        return edit.rotamer_list(m, chain, int(resseq), altloc=altloc, xmap=self.maps.get(self.refinement_map))

    @command("Set a residue's side chain to a named library rotamer.",
             {**RES, "name": ("string", "rotamer name, e.g. mt or m-85"), "altloc": ("string", ""), **MOL})
    def set_rotamer(self, chain, resseq, name, altloc="", mol=None):
        self._require_no_rsr()
        m = self.mol(mol)
        m.checkpoint()
        edit.apply_rotamer(m, chain, int(resseq), name, altloc=altloc)
        return self._edited(m, f"{chain}/{resseq}: rotamer {name}")

    @command("Mutate a residue (ideal side chain, old chi angles kept where possible).",
             {**RES, "resname": ("string", "new residue name, e.g. LEU"), **MOL})
    def mutate(self, chain, resseq, resname, mol=None):
        self._require_no_rsr()
        m = self.mol(mol)
        m.checkpoint()
        edit.mutate(m, chain, int(resseq), resname)
        return self._edited(m, f"{chain}/{resseq} -> {resname.upper()}")

    @command("Flip the peptide between resseq and resseq+1.", {**RES, **MOL})
    def flip_peptide(self, chain, resseq, mol=None):
        self._require_no_rsr()
        m = self.mol(mol)
        m.checkpoint()
        edit.flip_peptide(m, chain, int(resseq))
        return self._edited(m, f"flipped peptide {chain}/{resseq}-{int(resseq) + 1}")

    def _pos(self, x, y, z):
        return [x, y, z] if x is not None else list(self.view["center"])

    @command("Add a water at given coordinates (default: the screen centre, like Coot's 'place atom at pointer').",
             {"x": ("number", ""), "y": ("number", ""), "z": ("number", ""), **MOL})
    def add_water(self, x=None, y=None, z=None, mol=None):
        self._require_no_rsr()
        m = self.mol(mol)
        m.checkpoint()
        r = edit.add_single_atom(m, self._pos(x, y, z))
        self._edited(m, f"water {r['chain']}/{r['resseq']}")
        return r

    @command("Add an ion (single atom) at coordinates (default: screen centre).",
             {"element": ("string", "e.g. MG, NA, CL, ZN, CA, K"), "x": ("number", ""), "y": ("number", ""),
              "z": ("number", ""), **MOL})
    def add_ion(self, element, x=None, y=None, z=None, mol=None):
        self._require_no_rsr()
        m = self.mol(mol)
        m.checkpoint()
        el = element.upper()
        r = edit.add_single_atom(m, self._pos(x, y, z), resname=el, element=el.capitalize(), atom_name=el)
        self._edited(m, f"{el} {r['chain']}/{r['resseq']}")
        return r

    @command("Find candidate water sites: difference-map peaks at H-bond distance from N/O. With add=true they "
             "are added to the model.",
             {"map_id": ("integer", "difference map (default: first visible diff map)"),
              "sigma": ("number", "default 3.0"), "add": ("boolean", ""), **MOL})
    def find_waters(self, map_id=None, sigma=3.0, add=False, mol=None):
        self._require_no_rsr()
        m = self.mol(mol)
        if map_id is None:
            map_id = next((x.id for x in self.maps.values() if x.kind == "diff" and x.visible), None)
        if map_id is None:
            raise ValueError("no difference map")
        sites = edit.find_water_sites(m, self.maps[int(map_id)], sigma)
        if add and sites:
            m.checkpoint()
            for s in sites:
                edit.add_single_atom(m, s["xyz"])
            self._edited(m, f"added {len(sites)} waters")
        return sites

    @command("Add a ligand from a restraint CIF (needs coordinates in the CIF, e.g. from acedrg/grade/eLBOW) at "
             "coordinates or the screen centre. The CIF is also used for refinement restraints.",
             {"cif": ("string", "CIF path"), "comp_id": ("string", "component id if several"),
              "x": ("number", ""), "y": ("number", ""), "z": ("number", ""), **MOL})
    def add_ligand(self, cif, comp_id=None, x=None, y=None, z=None, mol=None):
        self._require_no_rsr()
        m = self.mol(mol)
        p = self._path(cif)
        m.checkpoint()
        r = edit.add_ligand(m, p, self._pos(x, y, z), comp_id)
        if p not in self.cif_paths:
            self.cif_paths.append(p)
        self._edited(m, f"ligand {r['comp_id']} {r['chain']}/{r['resseq']}")
        return r

    @command("Add a residue at the N or C terminus of a chain with ideal geometry.",
             {"chain": RES["chain"], "terminus": ("string", "N or C"), "resname": ("string", "default ALA"),
              "phi": ("number", "default -60"), "psi": ("number", "default -45"), **MOL})
    def add_residue(self, chain, terminus="C", resname="ALA", phi=-60.0, psi=-45.0, mol=None):
        self._require_no_rsr()
        m = self.mol(mol)
        m.checkpoint()
        r = edit.add_residue(m, chain, terminus, resname, float(phi), float(psi))
        self._edited(m, f"added {resname} {chain}/{r['resseq']}")
        return r

    @command("Renumber residues start..end of a chain by an offset.",
             {**RANGE, "offset": ("integer", ""), **MOL})
    def renumber(self, chain, start, end, offset, mol=None):
        self._require_no_rsr()
        m = self.mol(mol)
        m.checkpoint()
        n = edit.renumber(m, chain, int(start), int(end), int(offset))
        return self._edited(m, f"renumbered {n} residues of {chain} by {offset:+d}")

    @command("Delete residues, an atom, one altloc, or side chains.",
             {**RANGE, "atom": ("string", ""), "altloc": ("string", ""),
              "side_chain_only": ("boolean", ""), **MOL})
    def delete(self, chain, start, end=None, atom=None, altloc=None, side_chain_only=False, mol=None):
        self._require_no_rsr()
        m = self.mol(mol)
        m.checkpoint()
        n = edit.delete(m, chain, int(start), end and int(end), atom, altloc, side_chain_only)
        return self._edited(m, f"deleted {n} {'residues' if atom is None and not side_chain_only and not altloc else 'atoms'}")

    @command("Set occupancy for a residue range (optionally one altloc).",
             {**RANGE, "occupancy": ("number", ""), "altloc": ("string", ""), **MOL})
    def set_occupancy(self, chain, start, end, occupancy, altloc=None, mol=None):
        m = self.mol(mol)
        m.checkpoint()
        edit.set_occupancy(m, chain, int(start), int(end), float(occupancy), altloc)
        return self._edited(m, f"occupancy {occupancy} for {chain}/{start}-{end}")

    @command("Set isotropic B for a residue range.", {**RANGE, "b": ("number", ""), **MOL})
    def set_bfactor(self, chain, start, end, b, mol=None):
        m = self.mol(mol)
        m.checkpoint()
        edit.set_bfactor(m, chain, int(start), int(end), float(b))
        return self._edited(m, f"B={b} for {chain}/{start}-{end}")

    @command("Undo the last model change. While a refinement is open, this rejects it.", {**MOL})
    def undo(self, mol=None):
        if self.rsr is not None:
            return self.reject_refinement()
        m = self.mol(mol)
        ok = m.undo()
        if ok:
            self.push_model(m)
        return "undone" if ok else "nothing to undo"

    @command("Redo.", {**MOL})
    def redo(self, mol=None):
        self._require_no_rsr()
        m = self.mol(mol)
        ok = m.redo()
        if ok:
            self.push_model(m)
        return "redone" if ok else "nothing to redo"
