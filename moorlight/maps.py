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
    # phenix.fobs_minus_fobs_map, phenix.polder
    ("FoFo", "PHFc", "diff", "Fo-Fo"),
    ("mFo-DFc_polder", "PHImFo-DFc_polder", "diff", "polder mFo-DFc"),
    ("mFo-DFc_bias_omit", "PHImFo-DFc_bias_omit", "diff", "bias-omit mFo-DFc"),
    ("mFo-DFc_omit", "PHImFo-DFc_omit", "diff", "omit mFo-DFc"),
]

# TorchRef difference refinement, current layout (difference_data.mtz with dF, PHDELWT and FEXT): dF on the dark
# phases PHDELWT, weighted by W_Q or W_InVa ("F*W" = amplitudes F times weight column W), and FWT/PHWT there is
# the extrapolated light map, not 2mFo-DFc. Matched before PRESETS when the file has this layout.
TORCHREF_DIFFERENCE = [
    ("dF*W_Q", "PHDELWT", "diff", "light - dark (q-weighted)"),
    ("dF*W_InVa", "PHDELWT", "diff", "light - dark (W_InVa)"),
    ("dF", "PHDELWT", "diff", "light - dark (unweighted)"),
    ("FWT", "PHWT", "2fofc", "extrapolated light (2Fext-Fc)"),
    ("FC", "PHIC", "2fofc", "Fc (dark + light model)"),
]

DEFAULT_LEVEL = {"2fofc": 1.5, "diff": 3.0}
COLORS = {"2fofc": "#3d7fff", "diff": "#00e000"}
# Shown by default when present (one 2fofc-like and one diff-like, or the WDF map).
DEFAULT_VISIBLE = {"FWT", "2FOFCWT", "DELFWT", "FOFCWT", "WDF", "dF*W_Q", "FoFo", "mFo-DFc_polder"}


def split_weight(f):
    """'F*W' -> ('F', 'W'): amplitudes F weighted by column W. 'F' -> ('F', None)."""
    col, _, w = f.partition("*")
    return col.strip(), w.strip() or None


def weighted_f_phi(mtz, f, phi):
    """gemmi ComplexAsuData of F*W (see split_weight) with phases phi; reflections missing a value are left out."""
    col, w = split_weight(f)
    amp = np.array(mtz.column_with_label(col), dtype=float)
    if w:
        amp = amp * np.array(mtz.column_with_label(w), dtype=float)
    ph = np.deg2rad(np.array(mtz.column_with_label(phi), dtype=float))
    ok = np.isfinite(amp) & np.isfinite(ph)
    hkl = np.array(mtz.make_miller_array(), dtype=np.int32)[ok]
    return gemmi.ComplexAsuData(mtz.cell, mtz.spacegroup, hkl, (amp * np.exp(1j * ph))[ok].astype(np.complex64))


def map_from_mtz(mtz, f, phi, sample_rate=3.0, d_min=None):
    """FFT map of columns f (may be weighted, 'F*W') and phi; d_min leaves out reflections beyond it."""
    if not d_min and not split_weight(f)[1]:
        return mtz.transform_f_phi_to_map(f, phi, sample_rate=sample_rate)
    a = weighted_f_phi(mtz, f, phi) if split_weight(f)[1] else mtz.get_f_phi(f, phi)
    if d_min:
        a.value_array[a.make_d_array() < d_min] = 0
    return a.transform_f_phi_to_map(sample_rate=sample_rate)


def detect_columns(mtz: gemmi.Mtz):
    """Return list of (f, phi, kind, label) found in the MTZ, plus unpaired F/P columns."""
    labels = {c.label: c.type for c in mtz.columns}
    found = []
    used = set()
    presets = PRESETS
    if {"dF", "PHDELWT", "FEXT"} <= labels.keys():
        presets = TORCHREF_DIFFERENCE + PRESETS
    for f, phi, kind, label in presets:
        if f not in used and all(c in labels for c in split_weight(f) + (phi,) if c):
            found.append((f, phi, kind, label))
            used.add(f)
    # Generic fallback: any F column followed directly by a P column (cad-style pairs)
    cols = list(mtz.columns)
    for a, b in zip(cols, cols[1:]):
        if a.type == "F" and b.type == "P" and a.label not in used:
            kind, name = guess_kind(a.label), f"{a.label}/{b.label}"
            up = a.label.upper()
            for key, nice in (("BIAS_OMIT", "bias-omit"), ("POLDER", "polder"), ("OMIT", "omit")):
                if key in up:
                    name = f"{nice} {a.label}"
                    break
            found.append((a.label, b.label, kind, name))
            used.add(a.label)
    return found


def guess_kind(f_label):
    """'diff' for difference-map amplitudes (DELFWT, FOFCWT, mFo-DFc*, FoFo, DF...), else '2fofc'."""
    up = f_label.upper()
    if up.startswith("2"):
        return "2fofc"
    return "diff" if ("DEL" in up or up.startswith(("FOFC", "FOFO", "FO-FO", "DF", "WDF")) or "MFO-DFC" in up
                      or "POLDER" in up or "OMIT" in up) else "2fofc"


def is_density_label(f):
    """Amplitude labels known to be density (not difference) map coefficients."""
    up = f.upper()
    return up.startswith(("2", "FWT", "FEXT", "F-MODEL", "FMODEL")) or \
        up in ("F", "FO", "FP", "FOBS", "FC", "FCALC", "FC_ALL", "FWT_FILL")


def label_tells_kind(f):
    """True if the column label says what kind of map it is (else the values have to)."""
    return guess_kind(f) == "diff" or is_density_label(f)


def kind_from_values(arr):
    """'diff' or '2fofc' from map values: density maps have almost nothing below mean - 2.5 rms (< 0.004 % of the
    grid in 2mFo-DFc maps), difference maps have noise there (0.4-1.4 %)."""
    v = np.asarray(arr).ravel()
    v = v[:: max(1, v.size // 1_000_000)].astype(np.float64)
    return "2fofc" if np.mean(v < v.mean() - 2.5 * v.std()) < 0.001 else "diff"


def check_columns(mtz, f, phi):
    """Raise a readable error unless f is an amplitude-like and phi a phase column of the MTZ."""
    cols = {c.label: c.type for c in mtz.columns}
    missing = [c for c in split_weight(f) + (phi,) if c and c not in cols]
    if missing:
        raise ValueError(f"column(s) {missing} not in the MTZ; available: {list(cols)} "
                         "(note: MTZ labels cannot contain spaces)")
    if cols[phi] != "P":
        raise ValueError(f"column {phi} has type {cols[phi]}, expected a phase (P)")


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
        self.max_sigma = None      # largest |value - mean| / rms in the cell (known once the map is computed)
        self.kind_open = False     # kind only guessed from an unknown label: decided from the values when computed
        if grid is not None:
            self.grid = grid
        self.level_sigma = DEFAULT_LEVEL[kind]
        self.visible = False
        self.color = COLORS[kind]
        self.resolution = 2.0  # d_min of the map coefficients (set from the MTZ)
        self.d_min = None      # user resolution cut-off (None: all data)
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
        self.max_sigma = float(max(arr.max() - self.mean, self.mean - arr.min()) / self.rms) if self.rms else None
        if self.kind_open:
            self.kind_open = False
            kind = kind_from_values(arr)
            if kind != self.kind:
                self.kind, self.level_sigma, self.color = kind, DEFAULT_LEVEL[kind], COLORS[kind]

    @property
    def computed(self):
        return self._grid is not None

    @classmethod
    def from_mtz(cls, mtz, f, phi, kind, label, source, sample_rate=3.0, lazy=False, d_min=None):
        """d_min: leave out reflections beyond this resolution (to compare data sets at matched resolution)."""
        compute = lambda: map_from_mtz(mtz, f, phi, sample_rate, d_min)
        # map rms without the FFT (Parseval): sqrt(sum |F|^2 over P1 / V^2), good enough for display
        x = cls(label, None if lazy else compute(), kind, source, f, phi, compute=compute if lazy else None,
                rms=_rms_estimate(mtz, f, d_min) if lazy else None)
        x.resolution = max(float(mtz.resolution_high()), d_min or 0)
        x.d_min = d_min
        return x

    def info(self):
        return {
            "id": self.id, "name": self.name, "kind": self.kind, "source": self.source,
            "f": self.f, "phi": self.phi, "rms": self.rms, "mean": self.mean, "level_sigma": self.level_sigma,
            "level_abs": self.level_sigma * self.rms, "visible": self.visible,
            "color": self.color, "radius": self.radius, "resolution": round(self.resolution, 2), "d_min": self.d_min,
            "max_sigma": None if self.max_sigma is None else round(self.max_sigma, 2),
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

    def sigma_box(self, center, radius, max_n=96):
        """Values in sigma ((v - mean) / rms) on an orthogonal cube around center, at the map's grid spacing (coarser
        if the cube would have more than max_n points a side). Returns (arr float32 [i, j, k], origin, step);
        position of [i, j, k] = origin + step * (i, j, k)."""
        step = max(self.spacing(), 2 * radius / (max_n - 1))
        n = int(np.ceil(2 * radius / step)) + 1
        origin = np.asarray(center, dtype=float) - radius
        arr = np.zeros((n, n, n), dtype=np.float32)
        tr = gemmi.Transform()
        tr.mat.fromlist([[step, 0, 0], [0, step, 0], [0, 0, step]])
        tr.vec.fromlist(origin.tolist())
        self.grid.interpolate_values(arr, tr)
        arr -= self.mean
        arr /= self.rms
        return arr, origin, step

    def surfaces(self, center, radius, levels, carve_xyz=None, carve=None, step=0.33):
        """Smooth closed contour surfaces for figures: tricubic resampling at `step` Å, marching cubes at each level
        (sigma; difference maps at + and -). carve: keep only triangles whose corners all lie within `carve` Å of
        carve_xyz. Returns list of (sign, level, verts f32 (n,3), normals f32 (n,3), faces u32 (m,3))."""
        from skimage.measure import marching_cubes

        step = max(float(step), 2 * radius / 200)
        n = int(np.ceil(2 * radius / step)) + 1
        origin = np.asarray(center, dtype=float) - radius
        arr = np.zeros((n, n, n), dtype=np.float32)
        tr = gemmi.Transform()
        tr.mat.fromlist([[step, 0, 0], [0, step, 0], [0, 0, step]])
        tr.vec.fromlist(origin.tolist())
        self.grid.interpolate_values(arr, tr, order=3)
        arr = (arr - self.mean) / self.rms
        tree = None
        if carve and carve_xyz is not None and len(carve_xyz):
            from scipy.spatial import cKDTree
            tree = cKDTree(np.asarray(carve_xyz, dtype=float))
        out = []
        for sign in ([1, -1] if self.kind == "diff" else [1]):
            a = arr if sign > 0 else -arr
            for lev in levels:
                if a.max() < lev:
                    continue
                verts, faces, normals, _ = marching_cubes(a, float(lev), allow_degenerate=False)
                verts = verts * step + origin
                if tree is not None:
                    d, _ = tree.query(verts, distance_upper_bound=float(carve))
                    keep = np.isfinite(d)
                    faces = faces[keep[faces].all(1)]
                    used = np.unique(faces)
                    remap = np.full(len(verts), -1, np.int64)
                    remap[used] = np.arange(len(used))
                    verts, normals, faces = verts[used], normals[used], remap[faces]
                out.append((sign, float(lev), verts.astype(np.float32), (-normals).astype(np.float32),
                            faces.astype(np.uint32)))
        return out

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


def _rms_estimate(mtz, f, d_min=None):
    """Real-space rms of a map from its amplitudes (Parseval, F000 excluded)."""
    c, w = split_weight(f)
    col = np.array(mtz.column_with_label(c), dtype=float)
    if w:
        col = col * np.array(mtz.column_with_label(w), dtype=float)
    if d_min:
        col = col[np.array(mtz.make_d_array()) >= d_min]
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
