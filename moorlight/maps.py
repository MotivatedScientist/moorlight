"""Map handling: MTZ column detection, FFT maps, local contour meshes, peak search."""
from __future__ import annotations

import itertools
import struct
import json

import gemmi
import numpy as np

# (F, PHI, kind, label). kind "2fofc" = sigma map contoured at +level,
# kind "diff" = contoured at +level (green) and -level (red).
# Order matters: first match of each kind is shown by default.
PRESETS = [
    # TorchRef / phenix / refmac single-dataset refinement
    ("FWT", "PHWT", "2fofc", "2mFo-DFc"),
    ("2FOFCWT", "PH2FOFCWT", "2fofc", "2mFo-DFc"),
    ("DELFWT", "PHDELWT", "diff", "mFo-DFc"),
    ("FOFCWT", "PHFOFCWT", "diff", "mFo-DFc"),
    ("2FOFCWT_no_fill", "PH2FOFCWT_no_fill", "2fofc", "2mFo-DFc (no fill)"),
    ("2FOFCWT_fill", "PH2FOFCWT_fill", "2fofc", "2mFo-DFc (filled)"),
    ("FWT_no_fill", "PHWT_no_fill", "2fofc", "2mFo-DFc (no fill)"),
    ("F-model", "PHIF-model", "2fofc", "Fmodel"),
    ("FC", "PHIC", "2fofc", "Fc"),
    # TorchRef difference refinement (difference_data.mtz)
    ("WDF", "PHIC_dark", "diff", "WDF (weighted Fo-Fo, dark phases)"),
    ("DF", "PHIC_dark", "diff", "DF (Fo-Fo, dark phases)"),
    ("DFc", "PHIC_dark", "diff", "DFc (Fc-Fc, dark phases)"),
    ("mDFop-DFc", "PHIC_diff", "diff", "mDFo-DFc residual"),
    ("2mDFop-DFc", "PHIC_diff", "diff", "2mDFo-DFc"),
    ("2Fextc-Fc", "PHIC_light", "2fofc", "2Fext-Fc (c)"),
    ("Fextc-Fc", "PHIC_light", "diff", "Fext-Fc (c)"),
    ("2Fextp-Fc", "PHIC_light", "2fofc", "2Fext-Fc (p)"),
    ("Fextp-Fc", "PHIC_light", "diff", "Fext-Fc (p)"),
    ("2Fextb-Fc", "PHIC_light", "2fofc", "2Fext-Fc (b)"),
    ("Fextb-Fc", "PHIC_light", "diff", "Fext-Fc (b)"),
    ("F-model", "PH-model", "2fofc", "Fmodel"),
    ("Fc_dark", "PHIC_dark", "2fofc", "Fc dark"),
    ("Fc_light", "PHIC_light", "2fofc", "Fc light"),
]

DEFAULT_LEVEL = {"2fofc": 1.5, "diff": 3.0}
COLORS = {"2fofc": "#3d7fff", "diff": "#00e000"}
# Shown by default when present (one 2fofc-like and one diff-like, or the WDF map).
DEFAULT_VISIBLE = {"FWT", "2FOFCWT", "DELFWT", "FOFCWT", "WDF"}


def detect_columns(mtz: gemmi.Mtz):
    """Return list of (f, phi, kind, label) found in the MTZ, plus unpaired F/P columns."""
    labels = {c.label: c.type for c in mtz.columns}
    found = []
    used = set()
    for f, phi, kind, label in PRESETS:
        if f in labels and phi in labels:
            found.append((f, phi, kind, label))
            used.add(f)
    # Generic fallback: any F column followed directly by a P column (cad-style pairs)
    cols = list(mtz.columns)
    for a, b in zip(cols, cols[1:]):
        if a.type == "F" and b.type == "P" and a.label not in used:
            up = a.label.upper()
            kind = "diff" if ("DEL" in up or up.startswith("FOFC") or "MFO-DFC" in up and not up.startswith("2")) else "2fofc"
            found.append((a.label, b.label, kind, f"{a.label}/{b.label}"))
            used.add(a.label)
    return found


class XMap:
    _ids = itertools.count(1)

    def __init__(self, name, grid: gemmi.FloatGrid | None, kind="2fofc", source="", f=None, phi=None,
                 compute=None, rms=None):
        """grid=None with compute=callable makes the map lazy: the FFT runs when first needed."""
        self.id = next(self._ids)
        self.name = name
        self._grid = None
        self._compute = compute
        self.kind = kind
        self.source = source
        self.f, self.phi = f, phi
        self.mean, self.rms = 0.0, rms or 1.0
        if grid is not None:
            self.grid = grid
        self.level_sigma = DEFAULT_LEVEL[kind]
        self.visible = False
        self.color = COLORS[kind]
        self.resolution = 2.0  # d_min of the map coefficients (set from the MTZ)
        self.radius = 10.0  # contour box half-width in Å (Coot's "map radius")
        self.style = "mesh"

    @property
    def grid(self) -> gemmi.FloatGrid:
        if self._grid is None:
            self.grid = self._compute()
            self._compute = None
        return self._grid

    @grid.setter
    def grid(self, g):
        self._grid = g
        arr = np.array(g, copy=False)
        self.mean = float(arr.mean())
        self.rms = float(arr.std())

    @property
    def computed(self):
        return self._grid is not None

    @classmethod
    def from_mtz(cls, mtz, f, phi, kind, label, source, sample_rate=3.0, lazy=False):
        compute = lambda: mtz.transform_f_phi_to_map(f, phi, sample_rate=sample_rate)
        # map rms without the FFT (Parseval): sqrt(sum |F|^2 over P1 / V^2), good enough for display
        x = cls(label, None if lazy else compute(), kind, source, f, phi, compute=compute if lazy else None,
                rms=_rms_estimate(mtz, f) if lazy else None)
        x.resolution = float(mtz.resolution_high())
        return x

    def info(self):
        return {
            "id": self.id, "name": self.name, "kind": self.kind, "source": self.source,
            "f": self.f, "phi": self.phi, "rms": self.rms, "level_sigma": self.level_sigma,
            "level_abs": self.level_sigma * self.rms, "visible": self.visible,
            "color": self.color, "radius": self.radius,
        }

    def spacing(self):
        uc = self.grid.unit_cell
        return min(uc.a / self.grid.nu, uc.b / self.grid.nv, uc.c / self.grid.nw)

    def box(self, center, radius):
        """Resample the map on an orthogonal cube around center. Returns (arr, origin, step)."""
        step = self.spacing()
        n = int(np.ceil(2 * radius / step)) + 1
        origin = np.asarray(center, dtype=float) - radius
        arr = np.zeros((n, n, n), dtype=np.float32)
        tr = gemmi.Transform()
        tr.mat.fromlist([[step, 0, 0], [0, step, 0], [0, 0, step]])
        tr.vec.fromlist(origin.tolist())
        self.grid.interpolate_values(arr, tr)
        return arr, origin, step

    def value_at(self, xyz):
        return self.grid.interpolate_value(gemmi.Position(*xyz))

    def mesh(self, center, radius=None, level_sigma=None):
        """Marching-cubes contour(s) around center. Returns list of (sign, verts f32, edges u32)."""
        from skimage.measure import marching_cubes

        radius = radius or self.radius
        level_sigma = self.level_sigma if level_sigma is None else level_sigma
        arr, origin, step = self.box(center, radius)
        out = []
        signs = [1, -1] if self.kind == "diff" else [1]
        for sign in signs:
            level = self.mean + sign * level_sigma * self.rms
            a = arr if sign > 0 else -arr
            lev = level if sign > 0 else -level
            if a.max() < lev or a.min() > lev:
                out.append((sign, np.zeros((0, 3), np.float32), np.zeros((0,), np.uint32)))
                continue
            verts, faces, _, _ = marching_cubes(a, lev, allow_degenerate=False)
            verts = (verts * step + origin).astype(np.float32)
            # Coot-style chicken wire: draw triangle edges, de-duplicated.
            e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
            e.sort(axis=1)
            e = np.unique(e, axis=0).astype(np.uint32).ravel()
            out.append((sign, verts, e))
        return out

    def peaks(self, sigma=3.0, negative=False):
        """Local maxima above sigma (whole cell). Returns list of (xyz, height_sigma)."""
        from scipy.ndimage import maximum_filter

        arr = np.array(self.grid, copy=False)
        a = -arr if negative else arr
        thr = self.mean * (-1 if negative else 1) + sigma * self.rms
        mx = maximum_filter(a, size=3, mode="wrap")
        idx = np.argwhere((a == mx) & (a > thr))
        uc = self.grid.unit_cell
        res = []
        for u, v, w in idx:
            frac = gemmi.Fractional(u / self.grid.nu, v / self.grid.nv, w / self.grid.nw)
            pos = uc.orthogonalize(frac)
            res.append(((pos.x, pos.y, pos.z), float((a[u, v, w] - self.mean * (-1 if negative else 1)) / self.rms)))
        res.sort(key=lambda p: -p[1])
        return res


def _rms_estimate(mtz, f):
    """Real-space rms of a map from its amplitudes (Parseval, F000 excluded)."""
    col = np.array(mtz.column_with_label(f), dtype=float)
    col = col[np.isfinite(col)]
    hkl = np.array(mtz.make_miller_array())
    ops = mtz.spacegroup.operations()
    # multiplicity: each ASU reflection stands for |G|/|stabiliser| P1 reflections, x2 for Friedel mates
    n_p1 = len(ops) * 2
    return float(np.sqrt(n_p1 * (col ** 2).sum()) / mtz.cell.volume) if len(col) else 1.0


def pack_meshes(header: dict, meshes):
    """Binary payload: u32 header length, JSON header, then f32 vertices / u32 edges per mesh."""
    parts = []
    header = dict(header)
    header["meshes"] = []
    for sign, verts, edges in meshes:
        header["meshes"].append({"sign": sign, "nverts": int(len(verts)), "nedges": int(len(edges))})
        parts.append(np.ascontiguousarray(verts, np.float32).tobytes())
        parts.append(np.ascontiguousarray(edges, np.uint32).tobytes())
    hj = json.dumps(header).encode()
    pad = (-len(hj)) % 4
    hj += b" " * pad
    return struct.pack("<I", len(hj)) + hj + b"".join(parts)


class SymMapper:
    """Bring arbitrary positions to the symmetry/lattice image closest to a model."""

    def __init__(self, st: gemmi.Structure, xyz: np.ndarray, mask=None):
        from scipy.spatial import cKDTree

        self.cell = st.cell
        sg = st.find_spacegroup() or gemmi.SpaceGroup("P 1")
        self.ops = list(sg.operations())
        self.idx = np.arange(len(xyz)) if mask is None else np.nonzero(mask)[0]
        pts = xyz[self.idx]
        self.tree = cKDTree(pts)
        frac = np.array([self.cell.fractionalize(gemmi.Position(*p)).tolist() for p in pts[:: max(1, len(pts) // 500)]])
        self.centre = frac.mean(0) if len(frac) else np.zeros(3)
        self.shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], float)
        self.orth = np.array(self.cell.orth.mat.tolist())

    def nearest(self, xyz):
        """(position_near_model, distance, model_atom_index)."""
        f0 = gemmi.Fractional(*self.cell.fractionalize(gemmi.Position(*xyz)).tolist())
        cands = []
        for op in self.ops:
            f = np.array(op.apply_to_xyz(f0.tolist()))
            f -= np.round(f - self.centre)
            cands.append(f + self.shifts)
        cands = np.concatenate(cands) @ self.orth.T
        d, i = self.tree.query(cands)
        k = int(np.argmin(d))
        return cands[k], float(d[k]), int(self.idx[i[k]])
