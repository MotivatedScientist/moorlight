"""Shareable snapshot: one self-contained HTML viewer page with models, maps and preset slice planes.

No server or network is needed to view it: the page carries three.js and contours the maps itself, so it can be sent
as a file (e-mail, messenger, a paper's supplement) or published as a claude.ai artifact and opened on a phone.

    moorlight share init model.pdb maps.mtz -o spec.json --build view.html    # draft spec (and page) from files
    moorlight share spec.json -o view.html        # build the page
    moorlight share spec.json --check             # what the page shows: peaks, planes, values at atoms
    moorlight share spec.json --check --at "A/85 SG" --near "A/301" --radius 4

The spec (JSON; paths relative to the spec file):

    {"title": "Ligand 10 ps",                     # page name, two to four words
     "description": "one line shown under the title",
     "models": [{"path": "dark.pdb", "name": "dark"}],
     "maps": [{"path": "10ps.ccp4", "name": "10 ps light - dark", "kind": "diff",
               "units": "absolute", "unit_label": "SD", "level": 4.0, "visible": true}],
     "view": {"center": "A/401", "radius": 8, "orient": "face", "labels": ["A/401 N1"]},
     "planes": [{"name": "Ligand ring", "atoms": ["A/401 C1", "A/401 N1", "A/401 C2", "A/401 C3"],
                 "thickness": 0.5, "map": "10 ps light - dark"},
                {"name": "Zn site", "coordination": "A/301 ZN"}]}

See skills/moorlight-share/SKILL.md for every field.
"""
from __future__ import annotations

import argparse
import base64
import datetime
import gzip
import hashlib
import html
import itertools
import json
import os
import re
import subprocess
import sys

import gemmi
import numpy as np
from scipy.spatial import cKDTree

from .maps import DEFAULT_VISIBLE, SymMapper, detect_columns, check_columns, guess_kind, map_from_mtz
from .model import Molecule
from .session import parse_atom_spec

WEB = os.path.join(os.path.dirname(__file__), "web")
MAX_BYTES = 15 * 1024 * 1024    # artifact pages must stay at or below 16 MB
POINT_BUDGET = 8_000_000        # grid points stored for all maps together (~1.4 bytes each on the page)
THREE_VERSION = "0.170.0"       # vendored in web/vendor (MIT); written into every page
THREE_DIR = os.path.join(WEB, "vendor", f"three-{THREE_VERSION}")
THREE_ADDONS = ("lines/LineSegmentsGeometry.js", "lines/LineMaterial.js", "lines/LineSegments2.js",
                "objects/MarchingCubes.js")     # the addons the page uses, in dependency order


def three_bundle():
    """three.js and the addons the page uses as plain script text: each file runs in its own function scope and hands
    its exports on, so the page loads no modules and fetches nothing (from a file, an e-mail attachment or an
    artifact alike)."""
    with open(os.path.join(THREE_DIR, "LICENSE")) as fh:
        parts = [f"/* three.js {THREE_VERSION}, https://threejs.org\n{fh.read().strip()}\n*/"]
    with open(os.path.join(THREE_DIR, "three.module.min.js")) as fh:
        src = fh.read()
    m = re.search(r"export\s*\{([^}]*)\}\s*;?\s*$", src)
    if not m or len(re.findall(r"\bexport\b", src)) != 1:
        raise RuntimeError("three.module.min.js: expected one export statement at the end")
    names = [q.strip().split(" as ") for q in m.group(1).split(",") if q.strip()]
    exports = ",".join(f"{n[-1]}:{n[0]}" for n in names)
    parts.append(f"const THREE = (() => {{\n{src[:m.start()]}\nreturn Object.freeze({{{exports}}});\n}})();")
    for f in THREE_ADDONS:
        with open(os.path.join(THREE_DIR, f)) as fh:
            t = fh.read()
        t = re.sub(r"import\s*(\{[^}]*\})\s*from\s*'three';", r"const \1 = THREE;", t)
        t = re.sub(r"import\s*\{[^}]*\}\s*from\s*'\.\./[^']*';\n?", "", t)    # an earlier addon, already in scope
        ex = re.search(r"export\s*(\{[^}]*\})\s*;", t)
        body = t[:ex.start()] + t[ex.end():] if ex else t
        if not ex or re.search(r"^\s*(import|export)\b", body, re.M):
            raise RuntimeError(f"three.js addon {f}: unexpected import/export statements")
        parts.append(f"// three.js {THREE_VERSION} {f}\nconst {ex.group(1)} = (() => {{\n{body}\nreturn {ex.group(1)};\n}})();")
    return "\n".join(parts) + "\n"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_version():
    """Moorlight version and, in a git checkout, the commit (with -dirty for uncommitted changes)."""
    from . import __version__
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        out = subprocess.run(["git", "-C", here, "describe", "--always", "--dirty", "--abbrev=10"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return __version__, out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    d = here                                  # no git program (e.g. in a container): read the checkout
    while d != os.path.dirname(d) and not os.path.isdir(os.path.join(d, ".git")):
        d = os.path.dirname(d)
    g = os.path.join(d, ".git")
    try:
        with open(os.path.join(g, "HEAD")) as fh:
            head = fh.read().strip()
        if head.startswith("ref: "):
            ref = head[5:]
            if os.path.exists(os.path.join(g, ref)):
                with open(os.path.join(g, ref)) as fh:
                    return __version__, fh.read().strip()[:10]
            with open(os.path.join(g, "packed-refs")) as fh:
                for line in fh:
                    if line.strip().endswith(" " + ref):
                        return __version__, line.split()[0][:10]
            return __version__, None
        return __version__, head[:10]
    except OSError:
        return __version__, None
NEIGHBOUR_COLOR = "#8a9099"
WATERS = ("HOH", "WAT", "DOD", "H2O")
_TAGGED = re.compile(r"^(.*?)\s*\[([^\]]*)\]\s*$")


# --------------------------------------------------------- neighbour copies
def _shift_tag(t):
    parts = [("+" if v > 0 else "-") + (str(abs(v)) if abs(v) > 1 else "") + "abc"[a] for a, v in enumerate(t) if v]
    return "".join(parts)


class Neighbours:
    """Symmetry and lattice copies of a model's residues that come within `margin` Å of the model (the model itself
    excluded), as Coot draws symmetry atoms. Residues are copied whole, so the copies keep their bonds."""

    def __init__(self, mol: Molecule, margin=5.0):
        st = mol.st
        cell = st.cell
        sg = st.find_spacegroup() or gemmi.SpaceGroup("P 1")
        M = np.array(cell.orth.mat.tolist())
        Minv = np.array(cell.frac.mat.tolist())
        heavy = np.array([e != "H" for e in mol.elements])
        F = mol.xyz @ Minv.T
        heavy_idx = np.flatnonzero(heavy)
        keys = {}
        res = np.array([keys.setdefault(mol.info[i][:4], len(keys)) for i in heavy_idx])
        order = np.argsort(res, kind="stable")
        groups = np.split(heavy_idx[order], np.flatnonzero(np.diff(res[order])) + 1)
        tree = cKDTree(mol.xyz[heavy])
        lo, hi = mol.xyz[heavy].min(0) - margin, mol.xyz[heavy].max(0) + margin
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) @ Minv.T
        fmin, fmax = corners.min(0), corners.max(0)
        self.orig, self.tags, xyz = [], [], []
        for op in sg.operations():
            R, T = np.array(op.rot) / op.DEN, np.array(op.tran) / op.DEN
            Fo = F @ R.T + T
            ident = op.triplet() == "x,y,z"
            tmin = np.floor(fmin - Fo.max(0)).astype(int)
            tmax = np.ceil(fmax - Fo.min(0)).astype(int)
            for t in itertools.product(*(range(a, b + 1) for a, b in zip(tmin, tmax))):
                if ident and not any(t):
                    continue
                X = (Fo + np.array(t)) @ M.T
                Xh = X[heavy_idx]
                if (Xh.max(0) < lo).any() or (Xh.min(0) > hi).any():
                    continue
                close = np.isfinite(tree.query(Xh, distance_upper_bound=margin)[0])
                if not close.any():
                    continue
                tag = (("" if ident else op.triplet() + " ") + _shift_tag(t)).strip()
                for r in np.unique(res[close]):
                    g = groups[r]
                    self.orig.extend(g.tolist())
                    self.tags.extend([tag] * len(g))
                    xyz.append(X[g])
        self.xyz = np.concatenate(xyz) if xyz else np.zeros((0, 3))
        self.index = {(o, t): k for k, (o, t) in enumerate(zip(self.orig, self.tags))}
        # bonds of the model between atoms of the same copy
        pos = {}
        for k, (o, t) in enumerate(zip(self.orig, self.tags)):
            pos[(o, t)] = k
        bonds, orders = [], []
        for (i, j), order in zip(mol.bonds.tolist(), mol.orders.tolist()):
            for t in set(self.tags):
                a, b = pos.get((i, t)), pos.get((j, t))
                if a is not None and b is not None:
                    bonds.append((a, b))
                    orders.append(order)
        self.bonds, self.orders = bonds, orders
        self.mol = mol

    def to_json(self, number, name):
        m = self.mol
        atoms = []
        for o, t in zip(self.orig, self.tags):
            i = m.info[o]
            atoms.append(list(i[:6]) + [round(i[6], 2), round(i[7], 1), i[8], t])
        return {"id": number, "name": name, "xyz": np.round(self.xyz, 3).ravel().tolist(),
                "element": [m.elements[o] for o in self.orig], "atoms": atoms,
                "bonds": [v for b in self.bonds for v in b], "orders": self.orders, "visible": True,
                "color": NEIGHBOUR_COLOR, "mono": True, "hydrogens": False, "style": "lines", "neighbour": True}


# ------------------------------------------------------------------ atoms
class Atoms:
    """Atom-spec lookup over what the page draws.

    Molecule numbers follow the page: the spec's models (1, 2, ...), then their neighbour copies. A spec with a tag
    in brackets, 'S/14 O [+a]', names the copy of that atom one lattice vector a away."""

    def __init__(self, mols, neighbours):
        self.mols = mols
        self.nb = neighbours              # model number -> Neighbours (or absent)
        self.nb_number = {k: len(mols) + j + 1 for j, k in enumerate(sorted(neighbours))}

    def family(self, number):
        """Page molecule numbers of one model: the model and its neighbour copies."""
        if number <= len(self.mols):
            return {number, self.nb_number.get(number)} - {None}
        return {number} | {k for k, n in self.nb_number.items() if n == number}

    def by_name(self, spec):
        """'RET', 'ZN', 'AZM C1', '2#RET C13:B' -> the spec of that residue (or atom). With several residues of the
        name, the most buried one: the largest fraction of its atoms within 4 A of the model's polymer."""
        m_ = re.match(r"^(?:(\d+)\s*#\s*)?([A-Za-z0-9]{1,3})(?:\s+(\S+))?$", str(spec).strip())
        if not m_:
            raise ValueError(f"cannot read atom spec {spec!r}; use e.g. 'A/85 CA', 'A/85 LYS CA:B', '2#A/85 SG' or a "
                             "residue name such as 'RET' or 'ZN C1'")
        k, name, atom = int(m_.group(1) or 1), m_.group(2).upper(), m_.group(3)
        if not 1 <= k <= len(self.mols):
            raise ValueError(f"atom spec {spec!r}: no model {k} (there are {len(self.mols)})")
        m = self.mols[k - 1]
        found = {}
        for i, inf in enumerate(m.info):
            if inf[3].upper() == name and m.elements[i] != "H":
                found.setdefault(tuple(inf[:3]), []).append(i)
        if not found:
            names = sorted({inf[3] for inf in m.info if gemmi.find_tabulated_residue(inf[3]) is None
                            or not gemmi.find_tabulated_residue(inf[3]).is_amino_acid()})
            raise ValueError(f"atom spec {spec!r}: no residue named {name} in {m.name}; other residue names: "
                             + " ".join(names[:30]))
        key = next(iter(found))
        if len(found) > 1:
            poly = [i for i, inf in enumerate(m.info) if m.elements[i] != "H"
                    and (t := gemmi.find_tabulated_residue(inf[3])) is not None
                    and (t.is_amino_acid() or t.is_nucleic_acid())]
            if poly:
                tree = cKDTree(m.xyz[poly])
                key = max(found, key=lambda r: float(np.mean(tree.query(m.xyz[found[r]])[0] <= 4.0)))
        out = f"{k}#{key[0]}/{key[1]}{key[2]}"
        return out + (f" {atom}" if atom else "")

    def resolve(self, spec, first_conformer=True):
        """Atom spec -> (page molecule number, [atom indices], xyz array). A residue spec gives its heavy atoms."""
        tag = None
        if isinstance(spec, str):
            m_ = _TAGGED.match(spec)
            if m_:
                spec, tag = m_.group(1), m_.group(2).strip()
        try:
            d = parse_atom_spec(spec)
        except ValueError:
            if not isinstance(spec, str):
                raise
            d = parse_atom_spec(self.by_name(spec))
        k = d["mol"] or 1
        if not 1 <= k <= len(self.mols):
            raise ValueError(f"atom spec {spec!r}: no model {k} (there are {len(self.mols)})")
        m = self.mols[k - 1]
        res = [i for i, key in enumerate(m.keys) if key[0] == d["chain"] and key[1] == d["resseq"] and key[2] == d["icode"]]
        if not res:
            raise ValueError(f"atom spec {spec!r}: no residue {d['chain']}/{d['resseq']}{d['icode']} in {m.name}; "
                             f"residues: {self.residue_list(m)}")
        if d["atom"] and d["atom"] == m.info[res[0]][3] and not any(m.keys[i][3] == d["atom"] for i in res):
            d["atom"] = None                        # 'A/85 LYS': a residue name, not an atom
        idx = [i for i in res if (m.keys[i][3] == d["atom"] if d["atom"] else m.elements[i] != "H")
               and (d["altloc"] is None or m.keys[i][4] in (d["altloc"], ""))]
        if not idx:
            names = sorted({m.keys[i][3] for i in res})
            raise ValueError(f"atom spec {spec!r}: no such atom; atoms of that residue: {' '.join(names)}")
        if d["altloc"] is None and first_conformer:
            alts = sorted({m.keys[i][4] for i in idx} - {""})
            idx = [i for i in idx if m.keys[i][4] in ("", alts[0] if alts else "")]
        if not tag:
            return k, idx, m.xyz[idx]
        nb = self.nb.get(k)
        if nb is None:
            raise ValueError(f"atom spec {spec!r} [{tag}]: the page draws no neighbour copies of model {k}")
        out = [nb.index.get((i, tag)) for i in idx]
        if any(o is None for o in out):
            have = sorted({t for (o, t) in nb.index if o in idx})
            raise ValueError(f"atom spec {spec!r}: no copy [{tag}] on the page; copies drawn: "
                             + (", ".join(f"[{t}]" for t in have) or "none"))
        return self.nb_number[k], out, nb.xyz[out]

    def point(self, spec):
        """xyz list, an atom spec (residue: centroid) or a list of specs (centroid of all)."""
        if _is_xyz(spec):
            return np.asarray(spec, float)
        if isinstance(spec, (list, tuple)):
            return np.mean([self.point(s) for s in spec], axis=0)
        return self.resolve(spec)[2].mean(0)

    def xyz(self, specs):
        """All atom positions of a list of specs (residues give all their heavy atoms)."""
        out = [self.resolve(s)[2] for s in specs]
        return np.concatenate(out) if out else np.zeros((0, 3))

    def name(self, number, i):
        """Spec of an atom on the page, as the specs are written."""
        if number <= len(self.mols):
            a, tag, k = self.mols[number - 1].info[i], "", number
        else:
            k = next(kk for kk, nn in self.nb_number.items() if nn == number)
            nb = self.nb[k]
            a, tag = nb.mol.info[nb.orig[i]], nb.tags[i]
        spec = f"{a[0]}/{a[1]}{a[2]} {a[3]} {a[4]}" + (f":{a[5]}" if a[5] else "")
        if len(self.mols) > 1:
            spec = f"{k}#{spec}"
        return spec + (f" [{tag}]" if tag else "")

    def nearest(self):
        """f(xyz) -> (spec, distance) of the nearest heavy atom drawn on the page (models and neighbour copies)."""
        pts, ref = [], []
        for k, m in enumerate(self.mols, 1):
            for i, e in enumerate(m.elements):
                if e != "H":
                    pts.append(m.xyz[i])
                    ref.append((k, i))
        for k, nb in self.nb.items():
            for i in range(len(nb.orig)):
                pts.append(nb.xyz[i])
                ref.append((self.nb_number[k], i))
        tree = cKDTree(np.array(pts))

        def f(xyz):
            d, j = tree.query(xyz)
            return self.name(*ref[j]), float(d)
        return f

    def near_model(self):
        """f(xyz) -> the symmetry/lattice image of xyz next to the models (for listing each peak once)."""
        mappers = [SymMapper(m.st, m.xyz, np.array([e != "H" for e in m.elements])) for m in self.mols]

        def f(xyz):
            return np.asarray(min((sm.nearest(xyz) for sm in mappers), key=lambda r: r[1])[0])
        return f

    @staticmethod
    def residue_list(m, limit=40):
        seen = []
        for key, inf in zip(m.keys, m.info):
            r = f"{key[0]}/{key[1]}{key[2]} {inf[3]}"
            if r not in seen:
                seen.append(r)
        return ", ".join(seen[:limit]) + (" ..." if len(seen) > limit else "")


NON_METALS = ("C", "N", "O", "S", "P", "H", "SE", "CL", "BR", "I", "F", "B", "SI")


def coordination_cutoff(element):
    """Default metal-ligand cutoff (A): 2.6, or 2.8 for alkali and alkaline-earth ions (Ca-O up to ~2.7)."""
    return 2.8 if element.upper() in ("NA", "K", "RB", "CS", "CA", "SR", "BA") else 2.6


def _plane_rms(pts):
    """rms distance of points from their best plane."""
    c = pts - pts.mean(0)
    return float(np.sqrt(max(np.linalg.eigvalsh(c.T @ c)[0], 0.0) / len(pts)))


def _is_xyz(v):
    return isinstance(v, (list, tuple)) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v)


# ------------------------------------------------------------------- maps
def read_map(spec, base):
    """Map spec -> (gemmi.FloatGrid over the unit cell, kind or None). CCP4/MRC maps (a map that covers part of
    the cell has NaN elsewhere), or MTZ with columns 'F,PHI'."""
    p = _path(spec["path"], base)
    low = p.lower()
    if low.endswith((".ccp4", ".map", ".mrc")):
        g = gemmi.read_ccp4_map(p, setup=True).grid
        return g, spec.get("kind")
    if low.endswith(".mtz"):
        mtz = gemmi.read_mtz_file(p)
        if spec.get("columns"):
            f, phi = [c.strip() for c in spec["columns"].split(",")[:2]]
            check_columns(mtz, f, phi)
            kind = guess_kind(f)
        else:
            found = detect_columns(mtz)
            if not found:
                raise ValueError(f"{p}: no map columns found; give \"columns\": \"F,PHI\"")
            f, phi, kind, _ = found[0]
        sample = float(spec.get("sample_rate", 3.0))
        return map_from_mtz(mtz, f, phi, sample), spec.get("kind", kind)
    raise ValueError(f"{p}: unknown map format (use .ccp4/.map/.mrc or .mtz)")


def guess_map_kind(path, values):
    """'2fofc' or 'diff' from the file name (2mFo-DFc, FWT, emd_... / mFo-DFc, DELFWT, diff, minus, omit, ...),
    else from the values. Density maps have almost nothing below -2.5 rms (0.000-0.004 % of the grid in 2mFo-DFc
    maps); difference maps have noise there (0.4-1.4 %, also light - dark maps with strong positive features)."""
    name = os.path.basename(path).lower()
    if re.search(r"2m?fo|(?<!del)fwt|^emd", name):
        return "2fofc"
    if re.search(r"fo-?fc|mfo-?dfc|delfwt|diff|fo-?fo|minus|omit|polder|delta", name):
        return "diff"
    v = values[:: max(1, values.size // 1_000_000)].astype(np.float64)
    return "2fofc" if np.mean(v < v.mean() - 2.5 * v.std()) < 0.001 else "diff"


class PageMap:
    """A map as the page stores it: values on a grid with positions xyz = origin + G @ (i, j, k).

    native: the map's own grid over the unit cell, periodic, nothing resampled (values as in Coot).
    block: the part of that grid that covers the models and a margin, nothing resampled.
    box: resampled on an orthogonal box (when even the block is too large for the page)."""

    def __init__(self, spec, base, box_lo, box_hi, budget=POINT_BUDGET, cache=None):
        key = ("map", os.path.abspath(_path(spec["path"], base)), spec.get("columns"), spec.get("sample_rate"))
        if cache is None or key not in cache:
            got = read_map(spec, base)
            if cache is None:
                cache = {}
            cache[key] = got
        grid, kind = cache[key]
        whole = np.array(grid, copy=False)
        self.name = spec.get("name") or os.path.basename(spec["path"])
        finite = np.isfinite(whole)
        self.coverage = float(finite.mean())
        vals = whole[finite] if self.coverage < 1 else whole.ravel()
        if not vals.size:
            raise ValueError(f"map {self.name!r}: no values")
        self.mean, self.rms = float(vals.mean()), float(vals.std())
        if self.coverage < 1:                  # gemmi fills the cell outside the map's extent with NaN
            whole = np.where(finite, whole, np.float32(self.mean))
        kind = spec.get("kind") or kind or guess_map_kind(spec["path"], vals)
        self.kind = {"density": "2fofc", "em": "2fofc"}.get(kind, kind)
        if self.kind not in ("diff", "2fofc"):
            raise ValueError(f"map {self.name!r}: kind must be 'diff' or '2fofc' (or 'density')")
        self.units = spec.get("units", "sigma")
        if self.units not in ("sigma", "absolute"):
            raise ValueError(f"map {self.name!r}: units must be 'sigma' or 'absolute'")
        self.unit_label = spec.get("unit_label") or ("σ" if self.units == "sigma" else "")
        self.level = float(spec.get("level", 3.0 if self.kind == "diff" else 1.5))
        self.visible = bool(spec.get("visible", True))
        self.color = spec.get("color")
        self.source = os.path.basename(spec["path"])
        self.cell = np.array(grid.unit_cell.orth.mat.tolist())
        n_cell = np.array(whole.shape)
        g_cell = self.cell / n_cell                     # column a: one grid step along axis a
        lo, hi = np.asarray(box_lo, float), np.asarray(box_hi, float)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        g = np.linalg.solve(g_cell, corners.T).T        # the region in grid coordinates
        b_lo, b_hi = np.floor(g.min(0)).astype(int), np.ceil(g.max(0)).astype(int)
        block = int(np.prod(b_hi - b_lo + 1))
        mode = spec.get("grid", "auto")
        if mode == "auto":                              # smallest exact choice that fits the page
            if block < 0.5 * whole.size and block <= budget:
                mode = "block"
            elif whole.size <= budget:
                mode = "native"
            else:
                mode = "block" if block <= budget else "box"
        self.mode = mode
        self.filled = 0.0                               # fraction of stored points outside the map's extent
        if mode == "native":
            self.arr = np.array(whole, dtype=np.float32)
            self.n = n_cell
            self.G = g_cell
            self.origin = np.zeros(3)
            self.wrap = True
            self.filled = 1 - self.coverage
        elif mode == "block":
            idx = np.ix_(*[np.arange(a, b + 1) % n for a, b, n in zip(b_lo, b_hi, n_cell)])
            self.arr = np.array(whole[idx], dtype=np.float32)
            self.n = np.array(self.arr.shape)
            self.G = g_cell
            self.origin = g_cell @ b_lo
            self.wrap = False
            self.filled = float(1 - finite[idx].mean())
        elif mode == "box":
            uc = grid.unit_cell
            step = float(spec.get("step") or max(min(uc.a / grid.nu, uc.b / grid.nv, uc.c / grid.nw) / 2,
                                                 (np.prod(hi - lo) / budget) ** (1 / 3)))
            self.step, self.native_step = step, min(uc.a / grid.nu, uc.b / grid.nv, uc.c / grid.nw)
            self.n = np.maximum(2, np.ceil((hi - lo) / step).astype(int) + 1)
            self.arr = np.zeros(tuple(self.n), dtype=np.float32)
            tr = gemmi.Transform()
            tr.mat.fromlist([[step, 0, 0], [0, step, 0], [0, 0, step]])
            tr.vec.fromlist([float(v) for v in lo])
            if self.coverage < 1:
                grid = gemmi.FloatGrid(whole, grid.unit_cell, grid.spacegroup)
            grid.interpolate_values(self.arr, tr, order=3)   # tricubic: trilinear on a new grid flattens peaks
            self.G = np.eye(3) * step
            self.origin = lo
            self.wrap = False
        else:
            raise ValueError(f"map {self.name!r}: grid must be 'auto', 'native', 'block' or 'box'")
        # the page stores int16 steps of 1/100 of the map's rms (wider for maps with very high peaks); the values
        # here are rounded the same way, so every number --check prints is the one the page shows
        self.scale = max(self.rms / 100, float(np.abs(self.arr).max()) / 32000, 1e-12)
        self._raw = self.arr                   # peaks are found before rounding, which would make plateaus
        self.arr = (np.round(self.arr / self.scale) * self.scale).astype(np.float32)
        self.Ginv = np.linalg.inv(self.G)
        self._peaks = {}

    def to_level_units(self, v):
        """Map value -> the page's level units (σ of the whole map, or the map's own units)."""
        return (v - self.mean) / self.rms if self.units == "sigma" else v

    def value_at(self, xyz):
        """Trilinear interpolation on the stored grid, in level units (the page does the same); NaN outside a box."""
        f = self.Ginv @ (np.asarray(xyz, float) - self.origin)
        if not self.wrap and (np.any(f < 0) or np.any(f > self.n - 1)):
            return float("nan")
        i0 = np.floor(f).astype(int) if self.wrap else np.minimum(np.floor(f).astype(int), self.n - 2)
        t = f - i0
        v = 0.0
        for c in range(8):
            d = np.array([c & 1, (c >> 1) & 1, (c >> 2) & 1])
            idx = i0 + d
            if self.wrap:
                idx = idx % self.n
            v += np.prod(np.where(d, t, 1 - t)) * self.arr[tuple(idx)]
        return float(self.to_level_units(v))

    def image_near(self, xyz, c):
        """The lattice image of xyz closest to c (the stored grid is periodic only in native mode)."""
        if not self.wrap:
            return np.asarray(xyz, float)
        f = np.linalg.solve(self.cell, np.asarray(xyz, float) - c)
        return c + self.cell @ (f - np.round(f))

    def peaks(self, level, negative=False):
        """Grid points that are the extreme of their 3x3x3 neighbourhood beyond `level` (level units).
        One finder for every list: [(xyz, value)], strongest first."""
        key = (round(level, 4), negative)
        if key not in self._peaks:
            from scipy.ndimage import maximum_filter
            a = self.to_level_units(self._raw)
            a = -a if negative else a
            mx = maximum_filter(a, size=3, mode="wrap" if self.wrap else "nearest")
            hit = (a == mx) & (a > level)
            if not self.wrap:                  # the edge of a block or box is not a peak
                hit[[0, -1]] = hit[:, [0, -1]] = hit[:, :, [0, -1]] = False
            idx = np.argwhere(hit)
            val = self.to_level_units(self.arr[tuple(idx.T)]) * (-1 if negative else 1)   # the page's values
            order = np.argsort(-val)
            xyz = self.origin + idx[order] @ self.G.T
            self._peaks[key] = [(p, float(-v if negative else v)) for p, v in zip(xyz, val[order])]
        return self._peaks[key]

    def to_json(self):
        scale = self.scale
        q = np.ascontiguousarray(np.round(self.arr / scale).astype("<i2")).ravel().view(np.uint8).reshape(-1, 2)
        # high bytes first, then low bytes: the high bytes are mostly alike and compress ~15 % better apart
        data = base64.b64encode(gzip.compress(q[:, 1].tobytes() + q[:, 0].tobytes(), compresslevel=6, mtime=0)).decode()
        return {"name": self.name, "kind": self.kind, "units": self.units, "unit_label": self.unit_label,
                "level": self.level, "visible": self.visible, "color": self.color, "mean": self.mean, "rms": self.rms,
                "grid": self.mode, "n": [int(v) for v in self.n], "origin": [round(float(v), 4) for v in self.origin],
                "G": [round(float(v), 7) for v in self.G.ravel()], "Ginv": [round(float(v), 7) for v in self.Ginv.ravel()],
                "wrap": self.wrap, "scale": scale, "source": self.source, "data": data}


# ------------------------------------------------------------- orientation
def screen_axes(xyz, mode="face"):
    """Rows: screen x, y, z (towards the viewer) in model coordinates, from the atoms' principal axes.

    face: their plane faces the viewer, longest axis horizontal. edge: their plane is seen edge-on (its normal points
    up on screen), which shows out-of-plane motion."""
    c = xyz - xyz.mean(0)
    w, v = np.linalg.eigh(c.T @ c)          # ascending
    e1, e2, e3 = v[:, 2], v[:, 1], v[:, 0]  # longest, middle, shortest
    if mode == "edge":
        e2, e3 = e3, e2
    return np.array([e1, e2, np.cross(e1, e2)])


def _quat(R):
    """Quaternion [x, y, z, w] (three.js order) of a rotation matrix."""
    t = np.trace(R)
    if t > 0:
        s = 2 * np.sqrt(t + 1)
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, s / 4]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [s / 4, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2 * np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 1] + R[1, 0]) / s, s / 4, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s]
    else:
        s = 2 * np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, s / 4, (R[1, 0] - R[0, 1]) / s]
    q = np.array(q)
    return [round(float(v), 6) for v in q / np.linalg.norm(q)]


def _rot_from_quat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# ------------------------------------------------------------------ build
class Share:
    def __init__(self, spec, base, cache=None):
        """cache: a dict shared between Share objects of one run, so models, neighbours and maps are read once."""
        self.spec, self.base = spec, base
        cache = {} if cache is None else cache
        self.mols = []
        for k, ms in enumerate(spec.get("models") or []):
            key = ("model", k, os.path.abspath(_path(ms["path"], base)))
            if key not in cache:
                cache[key] = Molecule(_path(ms["path"], base), name=ms.get("name"))
            m = cache[key]
            m.name = ms.get("name") or os.path.basename(ms["path"])
            m.color, m.hydrogens = ms.get("color"), ms.get("hydrogens", False)
            m.style = ms.get("style")
            m.visible = ms.get("visible", True)
            m.neighbours = ms.get("neighbours", True)
            self.mols.append(m)
        if not self.mols:
            raise ValueError("the spec needs at least one model")
        lo, hi = self.region()
        margin = float(spec.get("neighbour_margin", 5.0))
        neighbours = {}
        for k, m in enumerate(self.mols, 1):
            if m.neighbours and m.st.cell.is_crystal():
                key = ("neighbours", k, id(m), margin)
                if key not in cache:
                    cache[key] = Neighbours(m, margin)
                nb = cache[key]
                if len(nb.orig):
                    neighbours[k] = nb
        self.atoms = Atoms(self.mols, neighbours)
        map_specs = spec.get("maps") or []
        self.maps = [PageMap(s, base, lo, hi, POINT_BUDGET / max(1, len(map_specs)), cache) for s in map_specs]
        names = [x.name for x in self.maps]
        if len(set(names)) != len(names):
            raise ValueError(f"map names must be unique: {names}")
        # the page opens on one view; older specs with "stops" open on their first stop
        first = spec.get("view") or (spec.get("stops") or [None])[0]
        self.view = self.stop(first, 0) if first else None
        self.stops = [self.view] if self.view else []
        self.planes = []
        for k, p in enumerate(spec.get("planes") or []):
            self.planes.append(self.plane(p, k))
            base = self.planes[-1]["name"]
            for o in p.get("copies") or []:      # parallel copies at these offsets (Å) along the normal
                sign = "+" if o > 0 else "\u2212"
                name = f"{base} {sign}{abs(o):g} \u00c5"
                self.planes.append(self.plane({**p, "name": name, "offset": float(p.get("offset", 0)) + float(o),
                                               "color": None}, k))
                self.planes[-1]["_copy"] = True

    def region(self):
        """Box for maps that are resampled (grid 'box'): all models plus a margin, or a cube around a centre."""
        r = self.spec.get("region") or {}
        if "center" in r:
            c, rad = self.point(r["center"]), float(r.get("radius", 12.0))
            return c - rad, c + rad
        xyz = np.concatenate([m.xyz for m in self.mols])
        margin = float(r.get("margin", 5.0))
        return xyz.min(0) - margin, xyz.max(0) + margin

    def point(self, spec):
        if hasattr(self, "atoms"):
            return self.atoms.point(spec)
        return Atoms(self.mols, {}).point(spec)

    def stop(self, s, k):
        where = "view" if k == 0 and not s.get("name") else f"stop {k + 1} ({s.get('name', '')!r})"
        try:
            center = self.atoms.point(s["center"])
            frame = s.get("frame")
            if frame is None and not _is_xyz(s["center"]):
                frame = s["center"] if isinstance(s["center"], (list, tuple)) else [s["center"]]
            mode = s.get("orient", "face")
            axes, orient, plane = None, None, None
            if isinstance(mode, (list, tuple)):
                orient = [float(v) for v in mode]
                axes = _rot_from_quat(orient)
            elif mode != "keep" and frame:
                pts = self.atoms.xyz(frame)
                if len(pts) < 3 and s.get("frame") is None and len(pts) == 1:
                    # a lone atom (a metal): use it and the atoms within 2.6 A, its coordination
                    num0, idx0, _ = self.atoms.resolve(frame[0])
                    fam = self.atoms.family(num0)
                    near = [(nb, i, xyz) for nb, i, xyz in self._drawn()
                            if nb in fam and 0.1 < np.linalg.norm(xyz - pts[0]) <= 2.6]
                    near_pts = [xyz for *_, xyz in self._same_conformer(self._alt(num0, idx0[0]), near)]
                    if len(near_pts) >= 2:
                        pts = np.vstack([pts] + near_pts)
                if len(pts) >= 3:
                    axes = screen_axes(pts, mode)
                    orient = _quat(axes)
                    plane = (pts.mean(0), axes[2] if mode == "face" else axes[1],
                             "towards the viewer" if mode == "face" else "up on screen")
            maps = {}
            for name, v in (s.get("maps") or {}).items():
                if name not in {x.name for x in self.maps}:
                    raise ValueError(f"no map named {name!r}; maps: {[x.name for x in self.maps]}")
                maps[name] = v
            labels = []
            for spec in s.get("labels") or []:
                number, idx, _ = self.atoms.resolve(spec)
                labels.extend([number, i] for i in idx)
            dists = []
            for a, b in s.get("distances") or []:
                na, ia, _ = self.atoms.resolve(a)
                nb, ib, _ = self.atoms.resolve(b)
                dists.append([[na, ia[0]], [nb, ib[0]]])
        except (ValueError, KeyError) as e:
            raise ValueError(f"{where}: {e}") from None
        radius = float(s.get("radius", 8.0))
        out = {"name": s.get("name") or ("start" if k == 0 else f"stop {k + 1}"), "text": s.get("text", ""),
               "center": [round(float(v), 3) for v in center], "orient": orient, "radius": radius,
               "slab": float(s.get("slab", radius)), "maps": maps, "labels": labels, "distances": dists,
               "_center_spec": s["center"], "_plane": plane}
        if "contour_radius" in s:
            out["contour_radius"] = float(s["contour_radius"])
        return out

    def _drawn(self):
        """(page molecule number, index, xyz) of every heavy atom drawn on the page."""
        for k, m in enumerate(self.mols, 1):
            for i, e in enumerate(m.elements):
                if e != "H":
                    yield k, i, m.xyz[i]
        for k, nb in self.atoms.nb.items():
            for i in range(len(nb.orig)):
                yield self.atoms.nb_number[k], i, nb.xyz[i]

    def _alt(self, number, i):
        """Altloc of an atom on the page ('' for none)."""
        if number <= len(self.mols):
            return self.mols[number - 1].info[i][5]
        nb = self.atoms.nb[next(kk for kk, nn in self.atoms.nb_number.items() if nn == number)]
        return nb.mol.info[nb.orig[i]][5]

    def _res_of(self, number, i):
        """Residue of an atom on the page; each neighbour copy of a residue is a residue of its own."""
        if number <= len(self.mols):
            return (number,) + tuple(self.mols[number - 1].info[i][:3])
        nb = self.atoms.nb[next(kk for kk, nn in self.atoms.nb_number.items() if nn == number)]
        return (number, nb.tags[i]) + tuple(nb.mol.info[nb.orig[i]][:3])

    def _same_conformer(self, alt, cands):
        """Atoms (number, index, ...) that can be present together with a centre atom of altloc `alt`: no altloc or the
        same one. A centre without altloc keeps each residue's first conformer, so two alternative ligands (e.g. a
        substrate in A and a product in B) are never counted as coordinating at the same time."""
        if alt:
            return [c for c in cands if self._alt(c[0], c[1]) in ("", alt)]
        first = {}
        for c in cands:
            a = self._alt(c[0], c[1])
            if a:
                r = self._res_of(c[0], c[1])
                first[r] = min(first.get(r, a), a)
        return [c for c in cands if self._alt(c[0], c[1]) in ("", first.get(self._res_of(c[0], c[1])))]

    def ligands_of(self, spec):
        """[(atom spec, distance)] of the atoms coordinating a lone metal or ion atom (its own model), nearest first;
        [] for other atoms."""
        number, idx, xyz = self.atoms.resolve(spec)
        el = self._element(number, idx[0]) if len(idx) == 1 else "C"
        if el.upper() in NON_METALS:
            return []
        cut = coordination_cutoff(el)
        fam = self.atoms.family(number)
        near = [(nb, i, x) for nb, i, x in self._drawn() if nb in fam and 0.1 < np.linalg.norm(x - xyz[0]) <= cut]
        out = [(self.atoms.name(nb, i), float(np.linalg.norm(x - xyz[0])))
               for nb, i, x in self._same_conformer(self._alt(number, idx[0]), near)]
        return sorted(out, key=lambda t: t[1])

    def plane(self, p, k):
        """Preset slice plane: the best plane through `atoms` (three or more), through an atom and the atoms within
        `cutoff` of it (`coordination`: a metal's coordination plane), or `center` with a `normal` (a vector, or two
        atom specs from the first to the second). `toward` (atom spec or point) turns the normal to that side;
        `axis` (two atom specs) sets its horizontal direction; `offset` moves it along the normal; `thickness`,
        `radius`, `map`, `level`, `color`, `visible`."""
        where = f"plane {k + 1} ({p.get('name', '')!r})"
        try:
            rms, pts, centre, ligands, shift = None, None, None, None, 0.0
            if p.get("coordination"):
                number, idx, xyz = self.atoms.resolve(p["coordination"])
                if len(idx) != 1:
                    raise ValueError("\"coordination\" is one atom, e.g. \"A/301 ZN\"")
                el = self._element(number, idx[0])     # a metal: its ligands; another atom: its bonded neighbours
                cut = float(p.get("cutoff") or (1.9 if el.upper() in NON_METALS else coordination_cutoff(el)))
                fam = self.atoms.family(number)          # another model's copy of the site is not a ligand
                ligands = [(nb, i, float(np.linalg.norm(x - xyz[0]))) for nb, i, x in self._drawn()
                           if nb in fam and 0.1 < np.linalg.norm(x - xyz[0]) <= cut]
                ligands = self._same_conformer(self._alt(number, idx[0]), ligands)
                if len(ligands) < 2:
                    raise ValueError(f"fewer than two atoms within {cut:g} A of {p['coordination']}; raise \"cutoff\"")
                ligands.sort(key=lambda t: t[2])
                lx = np.array([self._atom_xyz(nb, i) for nb, i, _ in ligands])
                use = list(range(len(ligands)))
                if len(ligands) >= 5 and p.get("equatorial", True):
                    # octahedral and higher: the four ligands that lie flattest with the metal (equatorial plane)
                    own_res = self._residue_key(number, idx[0])
                    same = [self._residue_key(nb, i) == own_res for nb, i, _ in ligands]
                    use = list(min(itertools.combinations(use, 4), key=lambda q: _plane_rms(np.vstack([xyz, lx[list(q)]]))
                                   + 0.1 * (4 - sum(same[j] for j in q))))
                pts = np.vstack([xyz, lx[use]])
                centre = xyz[0]
                lig_atoms = [ligands[j] for j in use]
                ligands = [(self.atoms.name(nb, i), d, j in use) for j, (nb, i, d) in enumerate(ligands)]
            elif p.get("perpendicular_to") is not None:
                # perpendicular to another plane (an earlier plane's name, or atoms), containing the line of `atoms`
                ref = p["perpendicular_to"]
                prev = next((q for q in self.planes if q["name"] == ref), None) if isinstance(ref, str) else None
                if prev is not None:
                    n_ref = np.array(prev["n"])
                else:
                    try:
                        rp = self.atoms.xyz(ref if isinstance(ref, list) else [t.strip() for t in str(ref).split(",")])
                    except ValueError:
                        raise ValueError(f"\"perpendicular_to\": {ref!r} is neither an earlier plane "
                                         f"({[q['name'] for q in self.planes]}) nor atoms") from None
                    if len(rp) < 3:
                        raise ValueError("\"perpendicular_to\" needs an earlier plane's name or three or more atoms")
                    n_ref = screen_axes(rp, "face")[2]
                line = self.atoms.xyz(p.get("atoms") or [])
                if len(line) < 2:
                    raise ValueError("a perpendicular plane needs \"atoms\": two or more along the line it contains")
                c0 = line - line.mean(0)
                dvec = np.linalg.eigh(c0.T @ c0)[1][:, 2] if len(line) > 2 else line[1] - line[0]
                dvec = dvec - n_ref * (dvec @ n_ref)
                if np.linalg.norm(dvec) < 1e-6:
                    raise ValueError("the atoms' line is along the other plane's normal; no plane is perpendicular "
                                     "to it and contains them")
                ex = dvec / np.linalg.norm(dvec)
                n = np.cross(n_ref, ex)
                n /= np.linalg.norm(n)
                ey = np.cross(n, ex)
                centre = self.atoms.point(p["center"]) if "center" in p else line.mean(0)
                h = (line - centre) @ n
                rms, shift = float(np.sqrt(np.mean(h ** 2))), 0.0
            elif p.get("atoms"):
                pts = self.atoms.xyz(p["atoms"])
                if len(pts) < 3:
                    raise ValueError("\"atoms\" needs three or more atoms (one atom: use \"coordination\")")
            if pts is not None and p.get("perpendicular_to") is None:
                ex, ey, n = screen_axes(pts, "face")
                if "center" in p:
                    centre = self.atoms.point(p["center"])
                elif centre is None:
                    centre = pts.mean(0)
                rms = _plane_rms(pts)
                shift = float((centre - pts.mean(0)) @ n)
            elif p.get("perpendicular_to") is not None:
                pass
            elif "center" in p and "normal" in p:
                centre = self.atoms.point(p["center"])
                nv = p["normal"]
                if _is_xyz(nv):
                    n = np.asarray(nv, float)
                elif isinstance(nv, (list, tuple)) and len(nv) == 2:
                    n = self.atoms.point(nv[1]) - self.atoms.point(nv[0])
                else:
                    raise ValueError("\"normal\" is a vector [x, y, z] or two atom specs")
                n = n / np.linalg.norm(n)
                ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
                ex = np.cross(ref, n)
                ex /= np.linalg.norm(ex)
                ey = np.cross(n, ex)
            else:
                raise ValueError("give \"atoms\" (three or more), \"coordination\" (one atom), or \"center\" and "
                                 "\"normal\"")
            if p.get("toward") is not None and (self.atoms.point(p["toward"]) - centre) @ n < 0:
                n, ey = -n, -ey                  # offsets and copies count towards that side
            if p.get("axis"):
                a, b = p["axis"]
                d = self.atoms.point(b) - self.atoms.point(a)
                d -= n * (d @ n)
                if np.linalg.norm(d) > 1e-6:
                    ex = d / np.linalg.norm(d)
                    ey = np.cross(n, ex)
            name = p.get("map")
            if name is not None and name not in {x.name for x in self.maps}:
                raise ValueError(f"no map named {name!r}; maps: {[x.name for x in self.maps]}")
        except (ValueError, KeyError) as e:
            raise ValueError(f"{where}: {e}") from None
        thick = float(p.get("thickness", 0.0))
        if ligands and "thickness" not in p:
            # tetrahedral and other non-planar sites: a slice thick enough to take in every ligand used
            h = np.abs((np.array([self._atom_xyz(nb, i) for nb, i, _ in lig_atoms]) - centre) @ n)
            if h.max() > 0.5:
                thick = float(min(4.0, np.ceil((2 * h.max() + 0.3) * 10) / 10))
        p0 = centre + n * float(p.get("offset", 0.0))
        own = []                                   # the atoms that define the plane: (name, xyz)
        try:
            if p.get("coordination"):
                number, idx, xyz = self.atoms.resolve(p["coordination"])
                own = [(self.atoms.name(number, idx[0]), xyz[0])] + [
                    (self.atoms.name(nb, i), self._atom_xyz(nb, i)) for nb, i, _ in lig_atoms]
            else:
                for sp in p.get("atoms") or []:
                    number, idx, xyz = self.atoms.resolve(sp)
                    own += [(self.atoms.name(number, i), x) for i, x in zip(idx, xyz)]
        except (ValueError, KeyError, NameError):
            own = []
        own = [(nm, float((x - p0) @ n), x) for nm, x in own]
        color = p.get("color")
        if isinstance(color, str):
            color = int(color.lstrip("#"), 16)
        r3 = lambda v: [round(float(t), 4) for t in v]
        return {"name": p.get("name") or f"Preset {k + 1}", "p0": r3(p0), "ex": r3(ex), "ey": r3(ey), "n": r3(n),
                "r": float(p.get("radius", self.spec.get("contour_radius", 10.0))),
                "thick": thick, "map": name, "level": p.get("level"), "color": color,
                "visible": bool(p.get("visible", True)), "_rms": rms, "_ligands": ligands, "_shift": shift,
                "_own": own}

    def payload(self):
        mols = []
        for k, m in enumerate(self.mols, 1):
            j = m.to_json()
            mols.append({"id": k, "name": m.name, "xyz": j["xyz"], "element": j["element"], "atoms": j["atoms"],
                         "bonds": j["bonds"], "orders": j["orders"], "visible": m.visible, "color": m.color,
                         "hydrogens": bool(m.hydrogens), "style": m.style or "sticks"})
        for k, nb in sorted(self.atoms.nb.items()):
            mols.append(nb.to_json(self.atoms.nb_number[k], f"{self.mols[k - 1].name}, neighbouring cells"))
        planes = [{k: v for k, v in pl.items() if not k.startswith("_")} for pl in self.planes]
        if self.view:
            start = {k: v for k, v in self.view.items() if not k.startswith("_")}
        else:
            c = np.concatenate([m.xyz for m in self.mols]).mean(0)
            start = {"name": "", "text": "", "center": [round(float(v), 3) for v in c], "orient": None,
                     "radius": 12.0, "slab": 12.0, "maps": {}, "labels": [], "distances": []}
        return {"title": self.spec.get("title", "Moorlight"), "description": self.spec.get("description", ""),
                "contour_radius": float(self.spec.get("contour_radius", 10.0)), "models": mols,
                "maps": [x.to_json() for x in self.maps], "planes": planes, "start": start, "about": self.about()}

    def about(self):
        """What the page was made from, for its About section: the input files with checksums, the tool and the
        build time, and the spec (file names without their directories). The recorded command line is left out: it
        holds absolute paths of the machine the page was built on, and stays in the .spec.json next to the page."""
        inputs, spec = [], json.loads(json.dumps(self.spec))
        spec.pop("command", None)
        for role, key in (("model", "models"), ("map", "maps")):
            for e in spec.get(key) or []:
                path = _path(e["path"], self.base)
                inputs.append({"role": role, "name": e.get("name") or os.path.basename(path),
                               "file": os.path.basename(path), "columns": e.get("columns"),
                               "bytes": os.path.getsize(path), "sha256": _sha256(path)})
                e["path"] = os.path.basename(path)
        version, commit = _source_version()
        built = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        return {"inputs": inputs, "moorlight": version, "commit": commit, "built": built, "three": THREE_VERSION,
                "spec": spec}

    def html(self, standalone=False):
        # `<` escaped as \u003c: no `</script>` or `<!--` can end or confuse the data block, whatever names the files hold
        data = json.dumps(_finite(self.payload()), separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
        with open(os.path.join(WEB, "share.html")) as fh:
            page = fh.read()
        with open(os.path.join(WEB, "share.js")) as fh:
            js = fh.read()
        js = re.sub(r'^import [^;]* from "three(/addons/[\w/]+\.js)?";\n', "", js, flags=re.M)   # in the bundle
        if re.search(r"^import\b", js, re.M):
            raise RuntimeError("share.js imports something the three.js bundle does not provide")
        script = three_bundle() + js
        if "</script" in script.lower():
            raise RuntimeError("page script contains '</script'")
        fill = {"TITLE": html.escape(self.spec.get("title", "Moorlight")),
                "DESCRIPTION": html.escape(self.spec.get("description", "")), "SCRIPT": script, "DATA": data}
        # one pass, so a title such as 'x {{DATA}}' stays text instead of pulling the data in as HTML
        page = re.sub(r"\{\{(TITLE|DESCRIPTION|SCRIPT|DATA)\}\}", lambda m: fill[m.group(1)], page)
        if standalone:
            page = ("<!doctype html>\n<html lang=\"en\"><head>\n"     # the page itself starts with <meta charset>
                    "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
                    + page + "\n</html>\n")
        return page

    # ------------------------------------------------------------- check
    def slice_peaks(self, pl, x, level=None):
        """[(xyz, value, h)] of map x's peaks (both signs for a difference map) within a plane's slice, strongest
        first; level defaults to the plane's level on its own map, else the map's level."""
        p0, n = np.array(pl["p0"]), np.array(pl["n"])
        ex, ey = np.array(pl["ex"]), np.array(pl["ey"])
        half = max(pl["thick"] / 2, 0.5)
        if level is None:
            level = float(pl["level"]) if pl["level"] is not None and pl["map"] == x.name else x.level
        found = []
        for neg in ((False, True) if x.kind == "diff" else (False,)):
            for xyz, v in x.peaks(level, negative=neg):
                q = x.image_near(xyz, p0)
                d = q - p0
                if abs(d @ n) <= half and abs(d @ ex) <= pl["r"] and abs(d @ ey) <= pl["r"]:
                    found.append((q, v, float(d @ n)))
        found.sort(key=lambda t: -abs(t[1]))
        return found

    def summary(self, page=None, size=None):
        """A few lines that say what the page shows, with numbers, for the reply that goes with it."""
        out = []
        p = out.append
        near = self.atoms.nearest()
        diff = next((x for x in self.maps if x.kind == "diff" and x.visible), None) or \
            next((x for x in self.maps if x.kind == "diff"), None)
        if page:
            p(f"{page}: {size / 1e6:.1f} MB, title {self.spec.get('title')!r}")
        v = self.view
        if v:
            c = v["_center_spec"]
            c = ", ".join(c) if isinstance(c, list) and not _is_xyz(c) else (c if isinstance(c, str) else "a point")
            p(f"opens on {c}, {2 * v['radius']:g} A across the screen" + (", face-on" if v["_plane"] else ""))
            if v["labels"]:
                p("  labels: " + ", ".join(self.atoms.name(nb, i) for nb, i in v["labels"]))
            if v["distances"]:
                p("  distances: " + ", ".join(
                    f"{self.atoms.name(*a)} - {self.atoms.name(*b)} "
                    f"{np.linalg.norm(self._atom_xyz(*a) - self._atom_xyz(*b)):.2f} A" for a, b in v["distances"]))
        for x in self.maps:
            u = x.unit_label or x.units
            p(f"map {x.name!r} [{'difference' if x.kind == 'diff' else 'density'}]: "
              f"{'shown' if x.visible else 'hidden'} at {'±' if x.kind == 'diff' else ''}{x.level:g} {u}"
              + (f"; rms {x.rms:.3g}" if x.units == "sigma" else f"; its rms is {x.rms:.3g} {u}"))
        coarse = [x for x in self.maps if x.mode == "box" and x.step > 1.2 * x.native_step]
        if coarse:
            x = coarse[0]
            p(f"note: too large for the page at the maps' own grid, so they are resampled at {x.step:.2f} A (their grid: "
              f"{x.native_step:.2f} A); values near peaks can be off by about half a {x.unit_label or x.units}. Add "
              "--around 15 (maps only around the site) for the exact values")
        if diff is not None and v:
            c = np.array(v["center"])
            r = max(6.0, min(v["radius"], 10.0))
            found = []
            for neg in (False, True):
                for xyz, val in diff.peaks(diff.level, negative=neg):
                    q = diff.image_near(xyz, c)
                    if np.linalg.norm(q - c) <= r:
                        found.append((abs(val), val, q))
            found.sort(key=lambda t: -t[0])
            u = diff.unit_label or diff.units
            p(f"strongest {diff.name!r} features within {r:g} A of the centre: "
              + ("; ".join(f"{val:+.1f} {u} {near(q)[1]:.1f} A from {near(q)[0]}" for _, val, q in found[:6])
                 or "none beyond the level"))
        if self.planes:
            p(f"planes (drawn in Slice mode; each shows the map in use):")
        for pl in self.planes:
            line = f"  {pl['name']}"
            if pl["_rms"] is not None:
                line += f": atoms {pl['_rms']:.2f} A rms off the plane"
            if pl["thick"]:
                line += f", {pl['thick']:g} A thick"
            if pl["_ligands"]:
                used = [f"{nm} {d:.2f}" for nm, d, u_ in pl["_ligands"] if u_]
                line += "; ligands " + ", ".join(used)
            hs = [h for _, h, _ in pl["_own"]] if not pl.get("_copy") else []
            if hs and max(abs(h) for h in hs) > 0.5:
                line += f"; its atoms lie {min(hs):+.1f} to {max(hs):+.1f} A off it"
            p(line)
            diffs = [m for m in self.maps if m.kind == "diff"]
            for x in diffs:
                u = x.unit_label or x.units
                tag = f"    {x.name}: " if len(diffs) > 1 else "    "
                parts = []
                if pl["_own"] and not pl.get("_copy"):
                    vals = np.array([x.value_at(xyz) for _, _, xyz in pl["_own"]])
                    parts.append(f"at its atoms rms {float(np.sqrt(np.mean(vals ** 2))):.1f}, "
                                 f"from {vals.min():+.1f} to {vals.max():+.1f} {u}")
                c = np.array(pl["p0"])
                pk = [t for t in self.slice_peaks(pl, x) if np.linalg.norm(t[0] - c) <= 6.0][:3]
                if pk:
                    parts.append("in the slice near its centre: " + ", ".join(
                        f"{val:+.1f} at {near(q)[0]}" for q, val, _ in pk))
                if parts:
                    p(tag + "; ".join(parts))
        return "\n".join(out)

    def _peak_lines(self, x, center, radius, level, near, plane=None, limit=8):
        """Peaks of one map within radius of center (their images next to it), as text lines."""
        u = x.unit_label or x.units
        lines = []
        for neg in ((False, True) if x.kind == "diff" else (False,)):
            found = []
            for xyz, v in x.peaks(level, negative=neg):
                p = x.image_near(xyz, center)
                if np.linalg.norm(p - center) <= radius:
                    found.append((p, v))
            head = f"    {'+' if not neg else '-'} beyond {'-' if neg else '+'}{level:g} {u}: {len(found)}"
            lines.append(head + (", strongest:" if found else ""))
            for p, v in found[:limit]:
                nm, d = near(p)
                h = ""
                if plane is not None:
                    h = f"  h {float((p - plane[0]) @ plane[1]):+.2f} A"
                lines.append(f"      {v:+7.2f} {u} at ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})  {d:.2f} A from {nm}{h}")
        return lines

    def report(self, at=(), near_specs=(), near_radius=5.0, n_peaks=12, stop_peaks=8, maps_only=(), frame=None,
               full=None):
        """What the page shows, in numbers. maps_only: names of the maps to list (default all); frame: atom specs whose
        plane gives h for the --near lists. With at or near_specs, only those sections unless full."""
        out = []
        p = out.append
        if full is None:
            full = not (at or near_specs)
        near = self.atoms.nearest()
        to_model = self.atoms.near_model()
        p("Peaks are grid points that are the extreme of their 3x3x3 neighbourhood; one finder serves every list. "
          "Values at atoms are trilinear on the stored grid, as the page shows them on a tap. Atom names with a tag in "
          "brackets ([+a], [-b+c], ...) are neighbour copies drawn on the page; use the same text in specs.")
        p("\nmodels: " + "; ".join(f"{k}# {m.name}: {len(m.keys)} atoms ({Atoms.residue_list(m, 12)})"
                                  for k, m in enumerate(self.mols, 1)))
        for k, nb in sorted(self.atoms.nb.items()):
            p(f"neighbour copies of model {k} (page molecule {self.atoms.nb_number[k]}): {len(nb.orig)} heavy atoms in residues "
              f"within {self.spec.get('neighbour_margin', 5.0)} A of the model; copies: "
              + ", ".join(f"[{t}]" for t in sorted(set(nb.tags))))
        for x in self.maps:
            if maps_only and x.name not in maps_only:
                continue
            u = x.unit_label or x.units
            vals = x.to_level_units(x.arr)
            grid = {"native": f"own grid {x.n.tolist()} over the cell, nothing resampled",
                    "block": f"block {x.n.tolist()} of its own grid around the models, nothing resampled",
                    "box": f"resampled on a box {x.n.tolist()} x {x.G[0, 0]:.2f} A (tricubic)"}[x.mode]
            if x.units == "sigma":     # the level on the map's own scale too
                on_scale = (f"±{x.level * x.rms:.3g}" if x.kind == "diff" else f"{x.mean + x.level * x.rms:.3g}") + " in map units"
            else:
                on_scale = f"{(x.level - (0 if x.kind == 'diff' else x.mean)) / x.rms:.3g} x rms"
            p(f"\nmap {x.name!r} ({x.source}, {x.kind}, {grid}): level {x.level:g} {u} = {on_scale}; whole map mean "
              f"{x.mean:.3g} rms {x.rms:.3g}; stored values {vals.min():.2f} to {vals.max():.2f} {u}")
            if x.filled > 0.001:
                p(f"  WARNING: the map file covers {100 * x.coverage:.0f} % of the cell; {100 * x.filled:.0f} % of the "
                  "stored grid lies outside it and shows the map's mean")
            for neg in ([False, True] if x.kind == "diff" else [False]):
                pk = x.peaks(x.level, negative=neg)
                listed = []          # symmetry/lattice copies of a peak are listed once, at the image next to the model
                for xyz, v in pk:
                    pos = to_model(xyz)
                    if any(np.linalg.norm(pos - q) < 0.5 for q, *_ in listed):
                        continue
                    listed.append((pos, v))
                    if len(listed) == n_peaks:
                        break
                p(f"  {'negative' if neg else 'positive'} peaks beyond {'-' if neg else '+'}{x.level:g} {u}: {len(pk)} on the "
                  f"stored grid" + (f"; strongest {len(listed)}, each once, next to the model:" if listed else ""))
                for pos, v in listed:
                    nm, d = near(pos)
                    p(f"    {v:+7.2f} {u} at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})  {d:.2f} A from {nm}")
        for s in self.stops:
            c = np.array(s["center"])
            plane = s["_plane"]
            p(f"\nstarting view: centre {s['_center_spec']!r} = ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}), radius {s['radius']:g}, "
              + (f"orientation from the frame atoms; h = height above their plane, positive {plane[2]}" if plane is not None
                 else "orientation " + ("given" if s["orient"] else "kept from the previous stop")))
            for x in self.maps:
                if not x.wrap and not np.isfinite(x.value_at(c)):
                    p(f"  WARNING: centre outside the box of map {x.name!r}")
            for number, i in [tuple(l) for l in s["labels"]]:
                xyz = self._atom_xyz(number, i)
                vals = ", ".join(f"{x.name} {x.value_at(xyz):+.2f}" for x in self.maps)
                h = f"; h {float((xyz - plane[0]) @ plane[1]):+.2f} A" if plane is not None else ""
                p(f"  label {self.atoms.name(number, i)}: {vals}{h}")
            for x in self.maps:
                if x.kind != "diff":
                    continue
                lv = s["maps"].get(x.name)
                shown = (lv is not False and lv is not None) if x.name in s["maps"] else x.visible
                lv = float(lv) if isinstance(lv, (int, float)) and not isinstance(lv, bool) else x.level
                if maps_only and x.name not in maps_only:
                    continue
                p(f"  {x.name} ({'shown' if shown else 'hidden'} at this stop, level {lv:g}), within {s['radius']:g} A:")
                out.extend(self._peak_lines(x, c, s["radius"], lv, near, plane, limit=stop_peaks))
        for pl in self.planes:
            p0, n = np.array(pl["p0"]), np.array(pl["n"])
            ex, ey = np.array(pl["ex"]), np.array(pl["ey"])
            half = max(0.5, pl["thick"] / 2)
            p(f"\nplane {pl['name']!r}: through ({p0[0]:.2f}, {p0[1]:.2f}, {p0[2]:.2f}), normal ({n[0]:.3f}, {n[1]:.3f}, "
              f"{n[2]:.3f}), {2 * pl['r']:g} A square, thickness {pl['thick']:g} A"
              + (f", atoms {pl['_rms']:.2f} A rms about their best plane" if pl["_rms"] is not None else "")
              + (f"; moved {pl['_shift']:+.2f} A along the normal to pass through the centre" if abs(pl["_shift"]) > 0.01
                 else ""))
            near_atoms = []
            for number, i, xyz in self._drawn():
                d = xyz - p0
                if abs(d @ n) <= half and abs(d @ ex) <= pl["r"] and abs(d @ ey) <= pl["r"]:
                    near_atoms.append((self.atoms.name(number, i), float(d @ n)))
            if pl["_ligands"]:
                inp = [f"{nm} {d:.2f} A" for nm, d, used in pl["_ligands"] if used]
                outp = [f"{nm} {d:.2f} A" for nm, d, used in pl["_ligands"] if not used]
                p("  coordination" + (" (equatorial plane)" if outp else "") + ": " + ", ".join(inp)
                  + ("; out of the plane: " + ", ".join(outp) if outp else ""))
            if pl["_own"]:
                p("  its own atoms (h): " + ", ".join(f"{nm} {h:+.2f}" for nm, h, _ in pl["_own"]))
            p(f"  atoms within {half:g} A of it: {len(near_atoms)}" + (": " + ", ".join(
                f"{nm} {h:+.2f}" for nm, h in near_atoms[:20]) + (" ..." if len(near_atoms) > 20 else "") if near_atoms else ""))
            for x in self.maps:
                if x.kind != "diff" or (pl["map"] and x.name != pl["map"]) or (maps_only and x.name not in maps_only):
                    continue
                lv = float(pl["level"]) if pl["level"] is not None else x.level
                u = x.unit_label or x.units
                for neg in (False, True):
                    found = []
                    for xyz, v in x.peaks(lv, negative=neg):
                        q = x.image_near(xyz, p0)
                        d = q - p0
                        if abs(d @ n) <= half and abs(d @ ex) <= pl["r"] and abs(d @ ey) <= pl["r"]:
                            found.append((q, v, float(d @ n)))
                    p(f"  {x.name}: {len(found)} {'negative' if neg else 'positive'} peaks beyond "
                      f"{'-' if neg else '+'}{lv:g} {u} in the slice" + (", strongest: " + "; ".join(
                          f"{v:+.1f} {near(q)[1]:.2f} A from {near(q)[0]} (h {h:+.2f})" for q, v, h in found[:stop_peaks])
                          if found else ""))
        if not full:
            del out[:]
        fplane = None
        if frame:
            pts = self.atoms.xyz(frame)
            if len(pts) >= 3:
                fplane = (pts.mean(0), screen_axes(pts, "face")[2], "on the side the plane normal points to")
        for spec in near_specs:
            c = self.atoms.point(_maybe_xyz(spec))
            p(f"\nnear {spec!r} = ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}), within {near_radius:g} A"
              + (f"; h = height above the plane of {frame}" if fplane is not None else "") + ":")
            p("  atoms by residue, nearest first (atom distance" + (", h" if fplane is not None else "") + "):")
            residues = {}
            for nm, d, xyz in self.atoms_near(c, near_radius, with_xyz=True):
                res, _, rest = nm.partition(" ")
                atom, _, tag = rest.partition(" ")
                residues.setdefault(f"{res} {tag}".strip(), []).append(
                    f"{atom} {d:.2f}" + (f" h {float((xyz - fplane[0]) @ fplane[1]):+.2f}" if fplane is not None else ""))
            for res, items in residues.items():
                p(f"    {res:<24} " + ", ".join(items))
            for x in self.maps:
                if not (maps_only and x.name not in maps_only):
                    p(f"  {x.name}:")
                    out.extend(self._peak_lines(x, c, near_radius, x.level, near, fplane, limit=max(12, stop_peaks)))
        for spec in (t.strip() for a in at for t in a.split(",") if t.strip()):
            number, idx, xyzs = self.atoms.resolve(spec, first_conformer=False)
            p(f"\nvalues at {spec!r}:")
            for i, xyz in zip(idx, xyzs):
                vals = ", ".join(f"{x.name} {x.value_at(xyz):+.2f}" for x in self.maps)
                p(f"  {self.atoms.name(number, i):<22} {vals}")
        return "\n".join(out)

    def atoms_near(self, c, r, with_xyz=False):
        """[(spec, distance)] (or (spec, distance, xyz)) of the heavy atoms drawn on the page within r of c,
        nearest first."""
        out = []
        for number, i, xyz in self._drawn():
            d = float(np.linalg.norm(xyz - c))
            if d <= r:
                out.append((self.atoms.name(number, i), d, xyz))
        out.sort(key=lambda t: t[1])
        return out if with_xyz else [t[:2] for t in out]

    def _residue_key(self, number, i):
        """(model, chain, number, icode) of an atom on the page; neighbour copies count as other residues."""
        if number > len(self.mols):
            return (number, i)
        return (number,) + tuple(self.mols[number - 1].info[i][:3])

    def _element(self, number, i):
        if number <= len(self.mols):
            return self.mols[number - 1].elements[i]
        nb = self.atoms.nb[next(kk for kk, nn in self.atoms.nb_number.items() if nn == number)]
        return nb.mol.elements[nb.orig[i]]

    def _atom_xyz(self, number, i):
        if number <= len(self.mols):
            return self.mols[number - 1].xyz[i]
        k = next(kk for kk, nn in self.atoms.nb_number.items() if nn == number)
        return self.atoms.nb[k].xyz[i]


def _maybe_xyz(spec):
    """'1.0, -2.5, 3' -> [1.0, -2.5, 3.0]; anything else unchanged."""
    try:
        v = [float(t) for t in spec.replace(",", " ").split()]
        return v if len(v) == 3 else spec
    except ValueError:
        return spec


def _path(p, base):
    p = os.path.expanduser(p)
    return p if os.path.isabs(p) else os.path.join(base, p)


def _finite(o):
    """NaN and infinities -> None (JSON.parse rejects them, which would leave the page blank)."""
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    return o


# ------------------------------------------------------------------- init
def _map_entries(path, name, base):
    """Spec entries for one map file. An MTZ gives every map a refinement program writes for viewing (2mFo-DFc and
    mFo-DFc), each with its columns; `name` then prefixes the map names."""
    stem = os.path.splitext(os.path.basename(path))[0]
    if not path.lower().endswith(".mtz"):
        return [{"path": path, "name": name or stem}]
    if ":" in name:                                  # file.mtz=name:F,PHI
        name, _, cols = name.partition(":")
        return [{"path": path, "columns": cols, "name": name or stem}]
    found = detect_columns(gemmi.read_mtz_file(_path(path, base)))
    if not found:
        raise ValueError(f"{path}: no map columns found; give them as {path}=name:F,PHI")
    shown = [f for f in found if f[0] in DEFAULT_VISIBLE] or found[:1]
    return [{"path": path, "columns": f"{f},{phi}", "kind": kind,
             "name": (f"{name} {label}" if len(shown) > 1 else name) if name else label}
            for f, phi, kind, label in shown]


def _residue_spec(sh, k, i, atom=False):
    a = sh.mols[k - 1].info[i]
    sp = f"{a[0]}/{a[1]}{a[2]}" + (f" {a[4]}" + (f":{a[5]}" if a[5] else "") if atom else "")
    return sp if len(sh.mols) == 1 else f"{k}#{sp}"


def _feature_planes(sh, x, thr, n, near=None):
    """Preset planes at the strongest features of difference map x: peaks of either sign beyond thr within 1.5 A of
    a model atom. A residue with more than five heavy atoms is a group of its own (the plane goes through its atoms
    around the strongest feature); smaller ones (metals, waters, ions) are grouped by name (a metal's plane is its
    coordination). near=(xyz, r): only features within r of xyz."""
    to_model = sh.atoms.near_model()
    heavy = [(k, i) for k, m in enumerate(sh.mols, 1) for i, e in enumerate(m.elements) if e != "H"]
    tree = cKDTree(np.array([sh.mols[k - 1].xyz[i] for k, i in heavy]))
    size = {}                                    # (model, residue) -> heavy atom names (altlocs once)
    for k, i in heavy:
        size.setdefault((k, tuple(sh.mols[k - 1].info[i][:4])), set()).add(sh.mols[k - 1].info[i][4])
    feats = []                                   # (|v|, v, model, atom index)
    for neg in (False, True):
        seen = []
        for xyz, v in x.peaks(thr, negative=neg):
            pos = to_model(xyz)
            if any(np.linalg.norm(pos - q) < 0.5 for q in seen):
                continue
            seen.append(pos)
            d, j = tree.query(pos)
            if d <= 1.5 and (near is None or np.linalg.norm(pos - near[0]) <= near[1]):
                feats.append((abs(v), v, *heavy[j]))
    feats.sort(key=lambda f: -f[0])
    groups = {}
    for f in feats:
        inf = sh.mols[f[2] - 1].info[f[3]]
        rkey = tuple(inf[:4])
        big = len(size[(f[2], rkey)]) > 5
        if big or inf[3] not in WATERS:        # the same residue in several models (dark, light) is one group
            groups.setdefault(rkey if big else ("name", inf[3]), []).append(f)
    ranked = sorted(groups.items(), key=lambda kv: -kv[1][0][0])[:n]
    ranked.sort(key=lambda kv: (kv[0][0] == "name", -kv[1][0][0]))   # residues of their own first
    planes = []
    for gkey, fs in ranked:
        top = fs[0]
        k, m = top[2], sh.mols[top[2] - 1]
        inf = m.info[top[3]]
        name = f"{inf[3]} {inf[1]}{inf[2]}"
        res_idx = [i for i, i2 in enumerate(m.info) if i2[:4] == inf[:4] and m.elements[i] != "H"
                   and i2[5] in ("", inf[5])]
        if len(res_idx) >= 3:
            hot = m.xyz[top[3]]
            part = [i for i in res_idx if np.linalg.norm(m.xyz[i] - hot) <= 2.6] if len(res_idx) > 12 else res_idx
            if len(part) >= 3:
                planes.append({"name": name, "atoms": [_residue_spec(sh, k, i, atom=True) for i in part],
                               "level": thr if top[0] > 3 * thr else x.level})
        elif sum(1 for *_, xyz in sh._drawn() if 0.1 < np.linalg.norm(xyz - m.xyz[top[3]]) <= 2.6) >= 2:
            planes.append({"name": f"{name} coordination", "coordination": _residue_spec(sh, k, top[3], atom=True),
                           "level": thr if top[0] > 3 * thr else x.level})
    return planes


def _ligand_planes(sh, n, near=None):
    """Preset planes through the ligands (residues that are neither polymer nor water), the most buried first (the
    fraction of their atoms within 4 A of the model's polymer; then the largest), so a bound chromophore or inhibitor
    comes before surface lipids and additives. Then the coordination of single-atom residues (metals, ions) that have
    two or more atoms within 2.6 A. A residue present in several models (dark and light) is used once.
    near=(xyz, r): only residues whose centre lies within r of xyz."""
    ligands, ions = [], []
    drawn = None
    seen = set()
    for k, m in enumerate(sh.mols, 1):
        residues, polymer = {}, []
        for i, (inf, e) in enumerate(zip(m.info, m.elements)):
            if e != "H" and inf[3] not in WATERS and inf[5] in ("", "A"):
                residues.setdefault(tuple(inf[:4]), []).append(i)
        for rkey, idx in residues.items():
            tab = gemmi.find_tabulated_residue(rkey[3])
            if tab is not None and (tab.is_amino_acid() or tab.is_nucleic_acid()):
                polymer.extend(idx)
        tree = cKDTree(m.xyz[polymer]) if polymer else None
        for rkey, idx in residues.items():
            tab = gemmi.find_tabulated_residue(rkey[3])
            if tab is not None and (tab.is_amino_acid() or tab.is_nucleic_acid()) or rkey in seen:
                continue
            if near is not None and np.linalg.norm(m.xyz[idx].mean(0) - near[0]) > near[1]:
                continue
            seen.add(rkey)
            name = f"{rkey[3]} {rkey[1]}{rkey[2]}"
            if len(idx) >= 3:
                buried = float(np.mean(tree.query(m.xyz[idx])[0] <= 4.0)) if tree is not None else 0.0
                ligands.append(((round(buried, 1), len(idx)), {"name": name, "atoms": [_residue_spec(sh, k, idx[0])]}))
            elif len(idx) == 1:
                drawn = drawn if drawn is not None else np.array([xyz for *_, xyz in sh._drawn()])
                if np.sum((np.linalg.norm(drawn - m.xyz[idx[0]], axis=1) > 0.1)
                          & (np.linalg.norm(drawn - m.xyz[idx[0]], axis=1) <= 2.6)) >= 2:
                    ions.append({"name": f"{name} coordination", "coordination": _residue_spec(sh, k, idx[0], atom=True)})
    return ([p for _, p in sorted(ligands, key=lambda t: (-t[0][0], -t[0][1]))] + ions)[:n]


PLANE_OPTIONS = ("copies", "toward", "thickness", "offset", "axis", "center", "normal", "radius", "map", "level",
                 "perpendicular-to", "cutoff", "equatorial", "color")


def _requested_plane(sh, text):
    """'NAME=SPEC, SPEC, ...[; option=value ...]' -> a plane. One atom gives its coordination plane, more give their
    best plane; with perpendicular-to=PLANE the atoms give a line that the plane contains, perpendicular to PLANE (an
    earlier plane's name, or atoms). Options: see PLANE_OPTIONS (copies=-1,1; toward=SPEC; thickness=2; ...)."""
    head, *opts = [t.strip() for t in text.split(";")]
    name, _, specs = head.partition("=") if "=" in head else ("", "", head)
    items = [t.strip() for t in specs.split(",") if t.strip()]
    if not items:
        raise ValueError(f"--plane {text!r}: give NAME=SPEC, SPEC, ... (atom or residue specs)")
    d = {"name": name.strip() or ", ".join(items)}
    extra = {}
    for o in opts:
        if not o:
            continue
        key, eq, val = o.partition("=")
        key, val = key.strip().lower(), val.strip().replace("\u00c5", "").replace("\u2212", "-").strip()
        if not eq or key not in PLANE_OPTIONS:
            raise ValueError(f"--plane {text!r}: cannot read {o!r}; options: " + ", ".join(f"{k}=..." for k in PLANE_OPTIONS))
        try:
            if key == "copies":
                extra[key] = [float(t) for t in val.split(",") if t.strip()]
            elif key in ("thickness", "offset", "radius", "level", "cutoff"):
                extra[key] = float(val)
            elif key == "equatorial":
                extra[key] = val.lower() in ("1", "true", "yes", "on")
            elif key in ("axis", "normal"):
                parts = [t.strip() for t in val.split(",") if t.strip()]
                extra[key] = [float(t) for t in parts] if key == "normal" and len(parts) == 3 and all(
                    re.fullmatch(r"[-+]?[\d.]+(e[-+]?\d+)?", t) for t in parts) else parts
            else:
                extra[key.replace("-", "_")] = val
        except ValueError:
            raise ValueError(f"--plane {text!r}: bad value in {o!r}") from None
    try:
        if "perpendicular_to" in extra:
            sh.atoms.xyz(items)
            d["atoms"] = items
        else:
            count = sum(len(sh.atoms.resolve(t)[1]) for t in items)
            d.update({"coordination": items[0]} if count == 1 else {"atoms": items})
    except ValueError as e:
        raise ValueError(f"--plane {text!r}: {e}") from None
    d.update(extra)
    return d


def _plane_anchor(sh, pl):
    """Centre of the atoms a plane entry is built from."""
    specs = list(pl.get("atoms") or []) + ([pl["coordination"]] if pl.get("coordination") else [])
    return sh.atoms.xyz(specs).mean(0)


def _is_primary_density(name):
    """Density maps worth showing first: not extrapolated, Fc or Fmodel maps."""
    return not re.search(r"(?i)extrapolated|\bfc\b|fmodel|\bfc \(", name)


def init_spec(models, maps, title=None, units="sigma", unit_label=None, level=None, min_peak=None, n_planes=None,
              planes=(), base=".", description="", center=None, radius=None, drop=(), cache=None, around=None):
    """A ready-to-build spec from files: a starting view of the whole structure and preset slice planes.

    Planes: the requested ones ('NAME=SPEC, SPEC, ...'), then up to n_planes automatic ones (default 5, or 0 when
    planes are requested): at the strongest features of the first difference map, or, without a difference map or
    features, through the ligands and the coordination of metals. With a center, automatic planes stay within 15 A of
    it; with `around` (maps stored only within that radius of the starting centre) they stay inside that region, so
    no plane is drawn where the page holds no map."""
    spec = {"title": title or "Structure and maps", "description": description or "", "models": [], "maps": [],
            "planes": []}
    for m in models:
        p, _, name = m.partition("=")
        spec["models"].append({"path": p, "name": name or os.path.splitext(os.path.basename(p))[0]})
    entries = []
    for m in maps:
        path, _, name = m.partition("=")
        entries.extend(_map_entries(path, name, base))
    names = [e["name"] for e in entries]
    for e in entries:
        if names.count(e["name"]) > 1:
            e["name"] = f"{os.path.splitext(os.path.basename(e['path']))[0]} {e['name']}"
        e["units"] = units
        if unit_label:
            e["unit_label"] = unit_label
    for d in drop:
        gone = _match_map([e["name"] for e in entries], d)
        entries = [e for e in entries if e["name"] != gone]
    spec["maps"] = entries
    sh = Share(spec, base, cache)
    any_diff = any(x.kind == "diff" for x in sh.maps)
    first_density = next((x for x in sh.maps if x.kind == "2fofc" and _is_primary_density(x.name)), None) or \
        next((x for x in sh.maps if x.kind == "2fofc"), None)
    first_diff = next((x for x in sh.maps if x.kind == "diff"), None)
    for e, x in zip(entries, sh.maps):
        e["kind"] = x.kind
        e["level"] = level if level is not None and (x.kind == "diff" or not any_diff) else x.level
        e["visible"] = x is first_density or x is first_diff      # one density and one difference map
        x.level = e["level"]
    diff = next((x for x in sh.maps if x.kind == "diff"), None)
    shown = diff or (sh.maps[0] if sh.maps else None)
    for text in planes:
        spec["planes"].append(_requested_plane(sh, text))
    n_auto = (0 if planes else 5) if n_planes is None else n_planes
    thr = float(min_peak if min_peak is not None else 1.5 * diff.level) if diff is not None else None

    def automatic(n, near=None):
        found = _feature_planes(sh, diff, thr, n, near) if diff is not None and n else []
        return found or (_ligand_planes(sh, n, near) if n else [])

    limit = (max(4.0, float(around) - 2.0) if around else 15.0)
    near = None
    if center:
        try:
            near = (sh.atoms.xyz([t.strip() for t in center.split(",") if t.strip()]).mean(0), limit)
        except ValueError as e:
            raise ValueError(f"--center: {e}") from None
    elif around and not planes and n_auto:
        first = automatic(1)                  # the page will open on the first plane: keep the others near it
        if first:
            near = (_plane_anchor(sh, first[0]), limit)
    auto = automatic(n_auto, near)
    spec["planes"] += auto
    names = [pl["name"] for pl in spec["planes"]]
    for pl in spec["planes"]:
        if names.count(pl["name"]) > 1:
            first = str((pl.get("atoms") or [pl.get("coordination") or ""])[0])
            k = int(first.split("#")[0]) if "#" in first else 1
            pl["name"] += f" ({spec['models'][k - 1]['name']})"
    if shown is not None:
        for pl in spec["planes"]:
            pl.setdefault("map", shown.name)
    # starting view: --center, else the requested planes, else the first automatic plane, else the whole structure
    focus = [t.strip() for t in center.split(",") if t.strip()] if center else []
    if not focus:
        for pl in (spec["planes"][:len(planes)] if planes else spec["planes"][:1]):
            focus += list(pl.get("atoms") or []) + ([pl["coordination"]] if pl.get("coordination") else [])
    if focus:
        try:
            pts = sh.atoms.xyz(focus)
        except ValueError as e:
            raise ValueError(f"--center: {e}") from None
        c = pts.mean(0)
        rad = radius or float(np.clip(np.linalg.norm(pts - c, axis=1).max() + 4.0, 8.0, 20.0))
        spec["view"] = {"center": focus, "radius": round(rad, 1)}
        if len(pts) == 1:                          # a metal: label it and its ligands, with distances
            lig = sh.ligands_of(focus[0])
            if lig:
                spec["view"]["labels"] = [focus[0]] + [nm for nm, _ in lig]
                spec["view"]["distances"] = [[focus[0], nm] for nm, _ in lig]
    else:
        allxyz = np.concatenate([m.xyz[[i for i, inf in enumerate(m.info) if inf[3] not in WATERS
                                        and m.elements[i] != "H"]] for m in sh.mols])
        c = allxyz.mean(0)
        rad = radius or float(np.percentile(np.linalg.norm(allxyz - c, axis=1), 90)) + 2.0
        spec["view"] = {"center": [round(float(v), 2) for v in c], "radius": round(rad, 1),
                        "orient": _quat(screen_axes(allxyz, "face"))}
    return spec


def _match_map(names, text):
    """The map name that text names: exactly, or as the one name that contains it (case ignored)."""
    if text in names:
        return text
    hits = [n for n in names if text.lower() in n.lower()]
    if len(hits) == 1:
        return hits[0]
    raise ValueError(f"no single map matches {text!r}; maps: {names}")


def init_main(argv):
    ap = argparse.ArgumentParser(prog="moorlight share init", description=
                                 "Write a ready-to-build share spec from model and map files: a starting view of the "
                                 "whole structure and preset slice planes, plus the --check report next to it. Then "
                                 "edit the spec if needed and run `moorlight share spec.json -o page.html`.")
    ap.add_argument("files", nargs="+", help="models (.pdb/.cif) and maps (.ccp4/.map/.mrc/.mtz). Name a file with "
                                             "path=name; an MTZ gives its 2mFo-DFc and mFo-DFc maps (path=name:F,PHI "
                                             "picks columns)")
    ap.add_argument("-o", "--out", required=True, help="spec to write (.json)")
    ap.add_argument("--title", help="page name, two to four words")
    ap.add_argument("--description", help="one line under the title")
    ap.add_argument("--plane", action="append", default=[], metavar="NAME=SPECS",
                    help="a preset plane: 'Ring=A/401 C1, A/401 N1, A/401 C2' (best plane through the atoms), "
                         "'Heme=A/201' (a residue's heavy atoms) or 'Zn site=A/301 ZN' (one atom: its coordination "
                         "plane). Repeatable")
    ap.add_argument("--planes", type=int, default=None, dest="planes",
                    help="automatic preset planes (default 5, or 0 when --plane is given)")
    ap.add_argument("--units", choices=["sigma", "absolute"], default="sigma")
    ap.add_argument("--unit-label", help="text after levels, e.g. SD for maps in noise SD (with --units absolute)")
    ap.add_argument("--level", type=float, help="contour level of the difference maps (default 3 sigma)")
    ap.add_argument("--min", type=float, dest="min_peak", help="automatic planes: features from this level on "
                                                                 "(default 1.5 x the difference map's level)")
    ap.add_argument("--build", metavar="PAGE", help="also build the page")
    a = ap.parse_args(argv)
    out_dir = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(out_dir, exist_ok=True)
    models, maps = [], []
    for f in a.files:
        path, eq, name = f.partition("=")
        if not os.path.exists(path):
            print(f"error: no file {path}", file=sys.stderr)
            return 1
        rel = os.path.relpath(os.path.abspath(path), out_dir) + (eq + name)
        (models if path.lower().endswith((".pdb", ".ent", ".cif", ".mmcif", ".pdb.gz", ".cif.gz")) else maps).append(rel)
    try:
        cache = {}
        spec = init_spec(models, maps, a.title, a.units, a.unit_label, a.level, a.min_peak, a.planes, a.plane, out_dir,
                         a.description, cache=cache)
        with open(a.out, "w") as fh:
            json.dump(spec, fh, indent=1, ensure_ascii=False)
        sh = Share(spec, out_dir, cache)
        check = os.path.splitext(a.out)[0] + ".check.txt"
        with open(check, "w") as fh:
            fh.write(sh.report())
    except (ValueError, OSError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"wrote {a.out}: {len(spec['maps'])} map(s) (" + ", ".join(f"{m['name']} [{m['kind']}]" for m in spec["maps"])
          + f"), {len(spec['planes'])} preset plane(s) (" + ", ".join(pl["name"] for pl in spec["planes"])
          + f"); and {check}")
    if a.build:
        page = sh.html()
        with open(a.build, "w") as fh:
            fh.write(page)
        print(f"wrote {a.build}: {len(page.encode()) / 1e6:.2f} MB")
    return 0


def list_model(path):
    """What a model holds, in spec form: chains with their residue ranges, ligands, metal and other ions with their
    coordination, waters and residues with alternate conformers."""
    st = gemmi.read_structure(path)
    st.setup_entities()
    out = [f"{path}: " + (f"{st.spacegroup_hm}, cell " + " ".join(f"{v:g}" for v in st.cell.parameters)
                          if st.cell.is_crystal() else "no crystal cell") + f"; {len(st)} model(s), the first listed"]
    ns = gemmi.NeighborSearch(st[0], st.cell, 5).populate()
    ligands, ions, waters, alts = {}, {}, [], []
    for ch in st[0]:
        poly = [r for r in ch if (t := gemmi.find_tabulated_residue(r.name)) is not None
                and (t.is_amino_acid() or t.is_nucleic_acid())]
        if poly:
            out.append(f"  chain {ch.name}: {len(poly)} polymer residues, {ch.name}/{poly[0].seqid.num}"
                       f"{poly[0].seqid.icode.strip()} {poly[0].name} to {ch.name}/{poly[-1].seqid.num}"
                       f"{poly[-1].seqid.icode.strip()} {poly[-1].name}")
        for r in ch:
            spec = f"{ch.name}/{r.seqid.num}{r.seqid.icode.strip()}"
            if any(a.altloc != "\0" for a in r):
                alts.append(spec)
            if r in poly:
                continue
            if r.name in WATERS:
                waters.append(spec)
                continue
            heavy = [a for a in r if a.element.name != "H"]
            if len(heavy) == 1:
                a = heavy[0]
                cut = coordination_cutoff(a.element.name)
                near = set()
                for m in ns.find_atoms(a.pos, "\0", radius=cut):   # marks sit in the cell; measure to the nearest image
                    d = st.cell.find_nearest_image(a.pos, m.pos, gemmi.Asu.Any).dist()
                    if m.to_cra(st[0]).atom.element.name != "H" and 0.1 < d <= cut:
                        near.add((m.chain_idx, m.residue_idx, m.atom_idx, round(d, 3)))
                names = []
                for ci, ri, ai, d in sorted(near, key=lambda t: t[3]):
                    cra = st[0][ci][ri]
                    names.append(f"{st[0][ci].name}/{cra.seqid.num}{cra.seqid.icode.strip()} {cra[ai].name} {d:.2f}")
                ions.setdefault((r.name, a.name, cut), []).append(f"{spec} ({len(near)}: " + ", ".join(names) + ")")
            else:
                ligands.setdefault((r.name, len(heavy)), []).append(spec)
    for (name, n), specs in ligands.items():
        out.append(f"  ligand {name} ({n} heavy atoms): " + ", ".join(specs[:40]) + (" ..." if len(specs) > 40 else ""))
    for (name, atom, cut), specs in ions.items():
        out.append(f"  {name} (atom {atom}; atoms within {cut:g} A): " + ", ".join(specs[:40])
                   + (" ..." if len(specs) > 40 else ""))
    out.append(f"  waters: {len(waters)}" + (f" ({', '.join(waters[:3])}{' ...' if len(waters) > 4 else ''}"
                                             f"{', ' + waters[-1] if len(waters) > 3 else ''})" if waters else ""))
    if alts:
        out.append(f"  alternate conformers in {len(alts)} residues: " + ", ".join(alts[:30])
                   + (" ..." if len(alts) > 30 else "") + " (specs without :B take the first conformer)")
    return "\n".join(out)


def list_atoms(path, text):
    """The atoms of the residues a spec or residue name names, one line per residue."""
    st = gemmi.read_structure(path)
    out = []
    want = text.strip()
    for ch in st[0]:
        for r in ch:
            spec = f"{ch.name}/{r.seqid.num}{r.seqid.icode.strip()}"
            if want.upper() in (r.name.upper(), spec) or want == f"{spec} {r.name}":
                out.append(f"  {spec} {r.name}: " + " ".join(
                    a.name + (f":{a.altloc}" if a.altloc != "\0" else "") for a in r if a.element.name != "H"))
    return "\n".join(out) or f"  no residue {text!r}"


def list_main(argv):
    ap = argparse.ArgumentParser(prog="moorlight share list", description=
                                 "List chains, residue ranges, ligands, metals (with their ligand atoms) and waters of "
                                 "a model, as atom specs; with residues, their atom names.")
    ap.add_argument("models", nargs="+", help="model files; after them, residues to list atoms of (A/501 or RET)")
    a = ap.parse_args(argv)
    files = [m for m in a.models if os.path.exists(m)]
    residues = [m for m in a.models if not os.path.exists(m)]
    try:
        for m in files:
            print(list_model(m) if not residues else f"{m}:")
            for r in residues:
                print(list_atoms(m, r))
    except (ValueError, OSError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


MODEL_EXT = (".pdb", ".ent", ".cif", ".mmcif", ".pdb.gz", ".cif.gz", ".ent.gz")
MAP_EXT = (".mtz", ".ccp4", ".map", ".mrc", ".map.gz", ".ccp4.gz")


MAX_DOWNLOAD = 500e6


def _fetch_pdb_redo(code, out_dir, name=""):
    """Download the PDB-REDO model and map coefficients of a PDB entry -> [model path, mtz path]."""
    import urllib.request
    code = code.lower()
    print(f"{code.upper()}: no such file here, so fetching the PDB-REDO entry {code.upper()} (pdb-redo.eu)", file=sys.stderr)
    d = os.path.join(out_dir, code)
    os.makedirs(d, exist_ok=True)
    got = []
    for ext in ("pdb", "mtz"):
        dst = os.path.join(d, f"{code}_final.{ext}")
        if not os.path.exists(dst):
            url = f"https://pdb-redo.eu/db/{code}/{code}_final.{ext}"
            try:
                with urllib.request.urlopen(url, timeout=60) as r, open(dst + ".part", "wb") as fh:
                    total = 0
                    while chunk := r.read(1 << 20):
                        total += len(chunk)
                        if total > MAX_DOWNLOAD:
                            raise OSError(f"more than {MAX_DOWNLOAD / 1e6:.0f} MB")
                        fh.write(chunk)
                os.replace(dst + ".part", dst)
            except OSError as e:
                raise ValueError(f"{code}: could not download {url} ({e}); give the model and map files instead")
        got.append(dst)
    return [f"{got[0]}={name or code.upper()}", got[1]]


def _expand_folder(d):
    """A refinement folder -> model and map arguments ('path=name'), for the layouts Moorlight knows:
    TorchRef difference refinement (*_dark.pdb, *_light.pdb, *difference_data.mtz, *_dark-sf.mtz; or D.pdb,
    L_refined.pdb, difference_data.mtz), TorchRef refinement (refined.pdb + refined.mtz), or one model and its maps."""
    names = sorted(os.listdir(d))
    j = lambda n: os.path.join(d, n)
    diff = [n for n in names if n.endswith("difference_data.mtz")]
    if diff:
        pre = diff[0][:-len("difference_data.mtz")]
        out = []
        for n, name in ((pre + "dark.pdb", "dark"), ("D.pdb", "dark"), (pre + "light.pdb", "light"),
                        ("L_refined.pdb", "light")):
            if n in names and not any(o.endswith("=" + name) for o in out):
                out.append(f"{j(n)}={name}")
        out.append(j(diff[0]))
        for n, name in ((pre + "dark-sf.mtz", "dark"), ("D.mtz", "dark")):
            if n in names:
                out.append(f"{j(n)}={name}")
                break
        return out
    if "refined.pdb" in names and "refined.mtz" in names:
        return [j("refined.pdb"), j("refined.mtz")]
    models = [n for n in names if n.lower().endswith(MODEL_EXT) and "-sf" not in n]
    maps = [n for n in names if n.lower().endswith(MAP_EXT)]
    if len(models) == 1 and maps:
        return [j(models[0])] + [j(n) for n in maps]
    raise ValueError(f"{d}: not a refinement folder I know (models {models}, maps {maps}); give the files")


def _spec_args(argv):
    """Specs may start with '-' (chain '-'): '--at -/1' -> '--at=-/1'."""
    fixed, k = [], 0
    while k < len(argv):
        if argv[k] in ("--at", "--near", "--frame", "--center", "--label", "--distance") and k + 1 < len(argv):
            fixed.append(f"{argv[k]}={argv[k + 1]}")
            k += 2
        else:
            fixed.append(argv[k])
            k += 1
    return fixed


def build_main(argv, prog="moorlight-share"):
    ap = argparse.ArgumentParser(prog=prog, description=
                                 "Build a viewer page from model and map files in one command, and print what it "
                                 "shows. Atom specs: 'A/85 CA', 'A/85' (residue), '2#A/85 SG' (model 2), or a residue "
                                 "name: 'RET', 'ZN', 'AZM C1' (the most buried one if there are several).",
                                 epilog="Other commands: `moorlight-share list MODEL [RESIDUE ...]` (what a model holds; "
                                        "a residue's atom names), `moorlight-share again PAGE.spec.json [options]` "
                                        "(the same page with more options), `moorlight-share SPEC.json --check` (the "
                                        "numbers of a spec).")
    ap.add_argument("inputs", nargs="+", help="model files (.pdb/.cif) and maps (.mtz/.ccp4/.map/.mrc); name one with "
                                              "path=name (file.mtz=name:F,PHI picks columns, F*W,PHI weights F); a "
                                              "refinement folder (TorchRef layouts); or a PDB id (PDB-REDO files)")
    ap.add_argument("-o", "--out", required=True, help="page to write (.html)")
    ap.add_argument("--title", help="page name, two to four words")
    ap.add_argument("--description", help="one line under the title")
    ap.add_argument("--center", metavar="SPECS", help="what the page opens on (comma-separated specs; a metal gets its "
                                                      "ligands labelled); default: the --plane atoms, else the strongest "
                                                      "difference feature, else the most buried ligand")
    ap.add_argument("--radius", type=float, help="A from the centre to the screen edge at the start")
    ap.add_argument("--plane", action="append", default=[], metavar="NAME=SPECS[; option=value]",
                    help="a preset slice plane (repeatable): 'Ring=A/401 C1, A/401 N1, A/401 C2' (best plane), "
                         "'Zn site=ZN' (one atom: its coordination plane), with options after ';': copies=-1,1  "
                         "toward=SPEC (side of +)  thickness=2  offset=1  axis=A,B  center=SPEC  "
                         "perpendicular-to=PLANE (then the atoms give a line in the plane)  map=NAME  level=4")
    ap.add_argument("--no-plane", action="append", default=[], metavar="NAME", help="remove a plane given earlier")
    ap.add_argument("--planes", type=int, help="automatic planes when no --plane is given (default 5)")
    ap.add_argument("--around", type=float, metavar="R", help="store the maps only within about R A of the "
                                                               "starting centre (much smaller files; e.g. 15)")
    ap.add_argument("--label", action="append", default=[], metavar="SPECS", help="atoms to label (comma-separated)")
    ap.add_argument("--distance", action="append", default=[], metavar="A, B", help="a distance to draw")
    ap.add_argument("--no-label", action="append", default=[], metavar="SPECS", help="remove labels given earlier")
    ap.add_argument("--no-distance", action="append", default=[], metavar="A, B", help="remove a distance given earlier")
    ap.add_argument("--show", action="append", default=[], metavar="MAP", help="show this map at the start")
    ap.add_argument("--hide", action="append", default=[], metavar="MAP", help="hide this map at the start")
    ap.add_argument("--drop", action="append", default=[], metavar="MAP", help="leave this map out of the page")
    ap.add_argument("--level", action="append", default=[], metavar="[MAP=]LEVEL",
                    help="contour level: LEVEL for the difference maps, MAP=LEVEL for one map (repeatable)")
    ap.add_argument("--units", choices=["sigma", "absolute"], default="sigma",
                    help="absolute: maps already in their own units, e.g. noise SD (with --unit-label SD)")
    ap.add_argument("--unit-label", help="text after levels, e.g. SD")
    ap.add_argument("--min", type=float, dest="min_peak", help="automatic planes: features from this level on")
    ap.add_argument("--standalone", action="store_true", help="a full HTML document to send as a file (e-mail, "
                                                              "messenger, supplement); without it, an artifact page")
    ap.add_argument("--at", action="append", default=[], metavar="SPECS", help="also print map values at these atoms")
    ap.add_argument("--near", action="append", default=[], metavar="SPEC", help="also print atoms and peaks around this")
    ap.add_argument("--near-radius", type=float, default=5.0)
    a = ap.parse_args(_spec_args(argv))
    out = os.path.abspath(a.out)
    out_dir = os.path.dirname(out)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(out)[0]
    try:
        files = []
        for f in a.inputs:
            path = f.partition("=")[0]
            if os.path.isdir(path):
                files += _expand_folder(path)
            elif not os.path.exists(path) and re.fullmatch(r"\d[A-Za-z0-9]{3}", path):
                files += _fetch_pdb_redo(path, out_dir, f.partition("=")[2])
            elif not os.path.exists(path):
                raise ValueError(f"no file {path}")
            else:
                files.append(f)
        models, maps = [], []
        for f in files:
            path, eq, name = f.partition("=")
            rel = os.path.relpath(os.path.abspath(path), out_dir) + (eq + name)
            (models if path.lower().endswith(MODEL_EXT) else maps).append(rel)
        diff_level = [float(v) for v in a.level if "=" not in v]
        planes = {}                       # by name: a later --plane replaces an earlier one in its place
        for t in a.plane:
            head = t.split(";")[0]
            planes[head.partition("=")[0].strip() if "=" in head else head.strip()] = t
        for n in a.no_plane:
            if planes.pop(n.strip(), None) is None:
                raise ValueError(f"--no-plane {n!r}: no such plane; planes: {list(planes)}")
        cache = {}
        spec = init_spec(models, maps, a.title, a.units, a.unit_label, diff_level[-1] if diff_level else None,
                         a.min_peak, a.planes, list(planes.values()), out_dir, a.description, a.center, a.radius,
                         a.drop, cache, a.around)
        if a.around:
            spec["region"] = {"center": spec["view"]["center"], "radius": a.around}
            sh0 = Share({**spec, "maps": [], "planes": []}, out_dir, cache)
            c0 = sh0.point(spec["view"]["center"])
            for pl in spec["planes"]:
                d = float(np.linalg.norm(_plane_anchor(sh0, pl) - c0))
                if d > a.around:
                    print(f"warning: plane {pl['name']!r} is {d:.0f} A from the centre, outside --around {a.around:g}; "
                          "it will show no map", file=sys.stderr)
        names = [m["name"] for m in spec["maps"]]
        for v in a.level:
            if "=" in v:
                n, _, lv = v.rpartition("=")
                spec["maps"][names.index(_match_map(names, n.strip()))]["level"] = float(lv)
        for n in a.show:
            spec["maps"][names.index(_match_map(names, n))]["visible"] = True
        for n in a.hide:
            spec["maps"][names.index(_match_map(names, n))]["visible"] = False
        view = spec["view"]
        for t in a.label:
            view.setdefault("labels", []).extend(x.strip() for x in t.split(",") if x.strip())
        for t in a.distance:
            pair = [x.strip() for x in t.split(",") if x.strip()]
            if len(pair) != 2:
                raise ValueError(f"--distance {t!r}: give two atoms, 'A, B'")
            view.setdefault("distances", []).append(pair)
        gone = {x.strip() for t in a.no_label for x in t.split(",") if x.strip()}
        if gone:
            view["labels"] = [x for x in view.get("labels", []) if x not in gone]
        for t in a.no_distance:
            pair = {x.strip() for x in t.split(",") if x.strip()}
            view["distances"] = [d for d in view.get("distances", []) if set(d) != pair]
        spec["command"] = [os.path.abspath(f.partition("=")[0]) + f.partition("=")[1] + f.partition("=")[2]
                           if os.path.exists(f.partition("=")[0]) else f for f in a.inputs]
        opts, k = [x for x in argv if x not in a.inputs], 0
        while k < len(opts):                       # the page path, absolute, so `again` works from anywhere
            if opts[k] in ("-o", "--out"):
                spec["command"] += ["-o", out]
                k += 2
            else:
                spec["command"] += ["-o", out] if opts[k].startswith("--out=") else [opts[k]]
                k += 1
        sh = Share(spec, out_dir, cache)
        page = sh.html(standalone=a.standalone)
        with open(out, "w") as fh:
            fh.write(page)
        with open(stem + ".spec.json", "w") as fh:
            json.dump(spec, fh, indent=1, ensure_ascii=False)
        with open(stem + ".check.txt", "w") as fh:
            fh.write(sh.report())
        size = len(page.encode())
        print(sh.summary(out, size))
        if a.at or a.near:
            print(sh.report(a.at, a.near, a.near_radius))
        print(f"also wrote {stem}.check.txt (every number) and {stem}.spec.json; to change the page, run "
              f"`moorlight-share again {stem}.spec.json` with more options (later options win; --plane adds)")
        if size > MAX_BYTES:
            print(f"warning: {size / 1e6:.1f} MB is more than an artifact page may hold (16 MB); --drop a map",
                  file=sys.stderr)
            return 2
        if a.standalone and size > 7e6:
            print(f"note: {size / 1e6:.1f} MB; e-mail encoding adds a third, so some mail servers (10 MB) will refuse "
                  "it; add --around 15 (maps only around the site), --drop a map, or send a link", file=sys.stderr)
    except (ValueError, OSError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


def again_main(argv):
    """Re-run the command recorded in a page's spec with more options."""
    if not argv or not argv[0].endswith(".json"):
        print("usage: moorlight-share again PAGE.spec.json [more options]", file=sys.stderr)
        return 1
    with open(argv[0]) as fh:
        spec = json.load(fh)
    if not spec.get("command"):
        print(f"error: {argv[0]} has no recorded command (it was not made by a build)", file=sys.stderr)
        return 1
    # the page next to this spec, whatever -o the recorded command names (a spec from elsewhere cannot write
    # anywhere else); an -o given now still wins
    cmd, k, own = [], 0, os.path.abspath(argv[0])[:-len(".spec.json")] + ".html" if argv[0].endswith(".spec.json") \
        else os.path.splitext(os.path.abspath(argv[0]))[0] + ".html"
    rec = spec["command"]
    while k < len(rec):
        if rec[k] in ("-o", "--out"):
            k += 2
        elif str(rec[k]).startswith("--out="):
            k += 1
        else:
            cmd.append(rec[k])
            k += 1
    print("again: moorlight-share " + " ".join(cmd + ["-o", own] + argv[1:]), file=sys.stderr)
    return build_main(cmd + ["-o", own] + argv[1:])


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "init":
        return init_main(argv[1:])
    if argv and argv[0] == "list":
        return list_main(argv[1:])
    if argv and argv[0] == "again":
        return again_main(argv[1:])
    if not any(x.endswith(".json") for x in argv if not x.startswith("-")):
        return build_main(argv)
    # specs may start with '-' (chain '-'): '--at -/1' -> '--at=-/1'
    fixed, k = [], 0
    while k < len(argv):
        if argv[k] in ("--at", "--near", "--frame") and k + 1 < len(argv):
            fixed.append(f"{argv[k]}={argv[k + 1]}")
            k += 2
        else:
            fixed.append(argv[k])
            k += 1
    ap = argparse.ArgumentParser(prog="moorlight share", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", help="share spec (JSON)")
    ap.add_argument("-o", "--out", help="output page (.html); default: next to the spec")
    ap.add_argument("--check", action="store_true", help="print what the page shows (peaks, planes, values at atoms); no page")
    ap.add_argument("--at", action="append", default=[], metavar="SPEC",
                    help="with --check: map values at these atoms or residues (comma-separated, or repeat --at)")
    ap.add_argument("--near", action="append", default=[], metavar="SPEC",
                    help="with --check: peaks of every difference map around an atom, residue or 'x,y,z' (repeatable)")
    ap.add_argument("--radius", "--near-radius", type=float, default=5.0, help="with --near: radius in A (default 5)")
    ap.add_argument("--peaks", type=int, default=12, help="with --check: peaks listed per map and sign")
    ap.add_argument("--stop-peaks", type=int, default=8, help="with --check: peaks listed per stop, map and sign")
    ap.add_argument("--map", action="append", default=[], dest="maps", metavar="NAME",
                    help="with --check: list only this map (repeatable)")
    ap.add_argument("--frame", metavar="SPECS", help="with --near: atoms (comma-separated) whose plane gives h")
    ap.add_argument("--standalone", action="store_true",
                    help="full HTML document, for sending as a file (e-mail, messenger, supplement); artifact pages "
                         "must not have it")
    a = ap.parse_args(fixed)
    with open(a.spec) as fh:
        spec = json.load(fh)
    try:
        sh = Share(spec, os.path.dirname(os.path.abspath(a.spec)))
        if a.check:
            frame = [t.strip() for t in a.frame.split(",")] if a.frame else None
            print(sh.report(a.at, a.near, a.radius, a.peaks, a.stop_peaks, a.maps, frame))
            return 0
    except (ValueError, OSError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    out = a.out or os.path.splitext(a.spec)[0] + ".html"
    page = sh.html(standalone=a.standalone)
    with open(out, "w") as fh:
        fh.write(page)
    size = len(page.encode())
    print(f"wrote {out}: {size / 1e6:.2f} MB, {len(sh.mols)} model(s) + {len(sh.atoms.nb)} neighbour set(s), "
          f"{len(sh.maps)} map(s) ({', '.join(x.mode for x in sh.maps)}), {len(sh.planes)} preset plane(s)")
    if size > MAX_BYTES:
        print(f"warning: larger than {MAX_BYTES / 1e6:.0f} MB; an artifact page must stay at or below 16 MB. "
              "Use fewer maps, or grid 'box' with a smaller region or a coarser step.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
