"""Session state and the command table shared by the UI, the HTTP API and the MCP tools."""
from __future__ import annotations

import asyncio
import base64
import glob
import inspect
import os
import threading
import time
import uuid

import gemmi
import numpy as np
from scipy.spatial import cKDTree

from . import edit
from .maps import XMap, detect_columns, DEFAULT_VISIBLE
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
             "and nearest residue, selected atoms, refinement in progress, fixed atoms, recent log.", view_only=True)
    def state(self):
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
        if self.rsr is not None:
            out["refinement"] = {"running": True, "moving_atoms": len(self.rsr.moving_mol),
                                 "altloc": self.rsr.altloc, **self.rsr.geometry_stats()}
        out["log"] = self.log[-15:]
        return out

    @staticmethod
    def _atom_label(m, i):
        k = m.info[i]
        return f"{k[0]}/{k[1]}{k[2]} {k[3]} {k[4]}{(':' + k[5]) if k[5] else ''}"

    @command("Take a screenshot of the 3D view. Returns the path of a PNG file you can Read.", view_only=True)
    def screenshot(self):
        data = self.ask_client("screenshot_request", {})
        png = base64.b64decode(data.split(",", 1)[1])
        d = os.path.join(self.workdir, ".moorlight", "screenshots")
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, time.strftime("view_%Y%m%d_%H%M%S.png"))
        with open(p, "wb") as fh:
            fh.write(png)
        return p

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
             "automatically. Optionally give explicit columns as 'F,PHI' strings.",
             {"path": ("string", "MTZ path"),
              "columns": ("array", "optional list of 'F,PHI' or 'F,PHI,diff' strings"),
              "show_all": ("boolean", "make all detected maps visible")})
    def load_mtz(self, path, columns=None, show_all=False):
        p = self._path(path)
        mtz = gemmi.read_mtz_file(p)
        if columns:
            found = []
            for c in columns:
                parts = [s.strip() for s in c.split(",")]
                kind = parts[2] if len(parts) > 2 else ("diff" if "DEL" in parts[0].upper() else "2fofc")
                found.append((parts[0], parts[1], kind, f"{parts[0]}/{parts[1]}"))
        else:
            found = detect_columns(mtz)
        if not found:
            raise ValueError(f"no map coefficient columns recognised in {p}: "
                             f"{[c.label for c in mtz.columns]}")
        new = []
        first_vis = not self.maps
        for f, phi, kind, label in found:
            vis = show_all or (first_vis and f in DEFAULT_VISIBLE)
            x = XMap.from_mtz(mtz, f, phi, kind, label, os.path.basename(p), self.sample_rate, lazy=not vis)
            x.visible = vis
            self.maps[x.id] = x
            new.append(x)
            if self.refinement_map is None and kind == "2fofc" and f not in ("F-model", "Fc_dark", "Fc_light"):
                self.refinement_map = x.id
            if self.diff_map is None and f in ("WDF", "DF"):
                self.diff_map = x.id
        self.push_maps()
        self.say(f"{os.path.basename(p)}: " + ", ".join(f"[{x.id}] {x.name}" for x in new))
        return [x.info() for x in new]

    @command("Load a CCP4/MRC map file.", {"path": ("string", "map path"),
                                            "difference": ("boolean", "contour at +/- level; omit to auto-detect "
                                                                  "(file name or symmetric +/- density)")})
    def load_map(self, path, difference=None):
        p = self._path(path)
        ccp4 = gemmi.read_ccp4_map(p, setup=True)
        if difference is None:
            difference = _looks_like_difference_map(p, ccp4)
        x = XMap(os.path.basename(p), ccp4.grid, "diff" if difference else "2fofc", os.path.basename(p))
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
        if not models:
            # e.g. <prefix>_refined.pdb; fall back to the only model in the folder
            models = sorted(n for n in names if n.endswith("refined.pdb")) or \
                sorted(n for n in names if n.endswith(".pdb"))[:1]
        if not models:
            models = sorted(n for n in names if n.endswith(("refined.cif", ".mmcif")))[:1]
        mtzs = [n for n in ("refined.mtz", "difference_data.mtz") if n in names] or \
            sorted(n for n in names if n.endswith(".mtz"))
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
        if "D.pdb" in ids and "L_refined.pdb" in ids:
            self.dark_model = ids["D.pdb"]
            self.set_active_molecule(ids["L_refined.pdb"])
            self.say("difference refinement: edits go to L_refined.pdb, D.pdb is the dark reference")
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
                        new = XMap.from_mtz(mtz, x.f, x.phi, x.kind, x.name, src, self.sample_rate)
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
    @command("Centre the view on a residue.", {**RES, "icode": ("string", "insertion code"), **MOL}, view_only=True)
    def goto(self, chain, resseq, icode="", mol=None):
        c = self.mol(mol).residue_center(chain, int(resseq), icode)
        self.set_center(*c)
        return c

    @command("Centre the view on coordinates.", {"x": ("number", ""), "y": ("number", ""), "z": ("number", "")},
             view_only=True)
    def set_center(self, x, y, z):
        self.view["center"] = [x, y, z]
        self.broadcast({"type": "view_cmd", "center": [x, y, z]})
        return [x, y, z]

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

    # ====================================================== map analysis
    @command("List strongest peaks (positive and negative) of a map, with the nearest atom of the active "
             "molecule. Useful for difference maps in time-resolved work.",
             {"map_id": ("integer", ""), "sigma": ("number", "threshold (default 3.5)"),
              "max_peaks": ("integer", "default 30"), "max_distance": ("number", "only peaks within this "
                                                                                 "distance of the model (default 4 Å)")},
             view_only=True)
    def peaks(self, map_id, sigma=3.5, max_peaks=30, max_distance=4.0):
        from .maps import SymMapper

        x = self.maps[int(map_id)]
        m = self.mol()
        sym = SymMapper(m.st, m.xyz, mask=np.array([e != "H" for e in m.elements]))
        out = []
        for neg in ([False, True] if x.kind == "diff" else [False]):
            for xyz, h in x.peaks(sigma, negative=neg):
                pos, d, ai = sym.nearest(xyz)
                if d > max_distance or any(np.linalg.norm(np.array(o["xyz"]) - pos) < 0.7 for o in out):
                    continue
                out.append({"height_sigma": round(-h if neg else h, 2), "xyz": [round(float(v), 2) for v in pos],
                            "nearest": self._atom_label(m, ai), "distance": round(d, 2)})
                if len(out) >= 4 * max_peaks:
                    break
        out.sort(key=lambda p: -abs(p["height_sigma"]))
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

    @command("Map value (in sigma) at each atom of a residue range, for every visible map.",
             {**RANGE}, view_only=True)
    def density_at_atoms(self, chain, start, end):
        m = self.mol()
        out = {}
        for x in self.maps.values():
            if not x.visible:
                continue
            vals = {}
            for i, k in enumerate(m.keys):
                if k[0] == chain and start <= k[1] <= end:
                    vals[self._atom_label(m, i)] = round((x.value_at(m.xyz[i]) - x.mean) / x.rms, 2)
            out[f"[{x.id}] {x.name}"] = vals
        return out

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
