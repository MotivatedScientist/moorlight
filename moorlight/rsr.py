"""Interactive real-space refinement using TorchRef geometry restraints.

For every refinement a small sub-model (moving residues + a fixed shell) is
written out and loaded into torchref.Model, so restraints always match the
current, edited model and ligand dictionaries. Moving atoms are optimised
against: TorchRef bond/angle/torsion/plane/chiral NLLs, a peptide omega term,
a simple non-bonded repulsion, the (sigma-scaled) density of the refinement
map, and harmonic "pull" springs for atoms being dragged in the viewer.
"""
from __future__ import annotations

import os
import tempfile
import threading
import time

import gemmi
import numpy as np
import torch
from scipy.spatial import cKDTree

DEG = np.pi / 180.0
_torchref_lock = threading.Lock()


_patched = False


def torchref_version():
    import importlib.metadata as md
    try:
        return md.version("torchref")
    except md.PackageNotFoundError:
        return "unknown"


def _is_v07():
    import importlib.util
    return importlib.util.find_spec("torchref.topology") is not None


def _patch_v05():
    """TorchRef <= 0.6: cache data files re-read on every build, tolerate zones without torsions."""
    global _patched
    if _patched:
        return
    import copy
    import functools
    import torchref.restraints.restraints as R
    import torchref.restraints.ramachandran as RA
    R.read_link_definitions = functools.lru_cache(maxsize=1)(R.read_link_definitions)
    RA.load_nll_surfaces = functools.lru_cache(maxsize=4)(RA.load_nll_surfaces)
    orig_cat = R.RestraintsNew.cat_dict

    def cat_dict(self):
        try:
            orig_cat(self)
        except KeyError:
            z = torch.zeros(0)
            self.restraints["torsion"]["all"] = {"indices": torch.zeros((0, 4), dtype=torch.long),
                                                 "references": z, "sigmas": z, "periods": z.long()}
            self._torsion_max_period = 1

    R.RestraintsNew.cat_dict = cat_dict
    orig_read = R.read_cif
    cache = {}

    def read_cif(path, *a, **k):
        key = (str(path), os.path.getmtime(path) if os.path.exists(str(path)) else 0, a, tuple(sorted(k.items())))
        if key not in cache:
            cache[key] = orig_read(path, *a, **k)
        return copy.deepcopy(cache[key])

    R.read_cif = read_cif
    _patched = True


def load_restraints(pdb_path, cif_paths=()):
    """Build TorchRef restraints for a PDB file.

    Returns (restraints dict ``R[type][origin][property]``, list of (chain, resseq, icode, name) per atom),
    for both the TorchRef 0.7 topology API and the older RestraintsNew API."""
    os.environ.setdefault("TORCHREF_DEVICE", "cpu")
    from torchref.model.model import Model
    present = {r.name for ch in gemmi.read_structure(pdb_path)[0] for r in ch}
    use = [c for c in cif_paths if os.path.splitext(os.path.basename(c))[0] in present
           or not os.path.basename(c)[:-4].isupper()]
    with _torchref_lock:
        if _is_v07():
            m = Model(verbose=0, device=torch.device("cpu"), cif_path=use or None)
            m.load_pdb(pdb_path)
            rr = m.restraints.restraints
            c = m.ctx.topology.columns()
            atoms = [(str(ch), int(n), str(ic or "").strip(), str(nm))
                     for ch, n, ic, nm in zip(c["chain"], c["resseq"], c["icode"], c["name"])]
            if "all" not in rr.get("torsion", {}):
                z = torch.zeros(0)
                rr.setdefault("torsion", {})["all"] = {"indices": torch.zeros((0, 4), dtype=torch.long),
                                                      "references": z, "sigmas": z, "periods": z.long()}
            return rr, atoms
        _patch_v05()
        m = Model(verbose=0, device=torch.device("cpu"))
        m.load_pdb(pdb_path)
        if use:
            m.set_restraints_cif(use)
        r = m.restraints
        r.cat_dict()
        df = m.pdb
        atoms = [(str(ch), int(n), str(ic or "").strip(), str(nm))
                 for ch, n, ic, nm in zip(df.chainid, df.resseq, df.icode.fillna(""), df.name)]
        return r.restraints, atoms


class RSRJob:
    """One refinement 'zone'. Runs in a background thread until accepted/rejected."""

    def __init__(self, mol, xmap, moving_idx, *, weight=60.0, altloc=None, cif_paths=(),
                 shell=6.0, on_update=None, torsions=True, nonbonded=False,
                 diff_map=None, diff_weight=60.0, dark_mol=None):
        """xmap: map whose density atoms are pulled into (None or weight 0: off).
        diff_map: difference map for the real-space difference target (None: off); the dark
        reference comes from dark_mol, else other altlocs of the same residue, else start coordinates.
        nonbonded: non-bonded repulsion between atoms (off lets atoms be pulled on top of each other)."""
        self.mol, self.xmap = mol, xmap
        self.weight = float(weight)
        self.use_nonbonded = bool(nonbonded)
        self.diff_map, self.diff_weight, self.dark_mol = diff_map, float(diff_weight), dark_mol
        self.on_update = on_update
        self.pulls: dict[int, np.ndarray] = {}  # mol atom index -> target xyz
        self.lock = threading.Lock()
        self.running = False
        self.done = False
        self.status = {}
        self.use_torsions = torsions
        self.pull_k = 3000.0
        moving_idx = [i for i in moving_idx if mol.keys[i] not in mol.fixed and mol.elements[i] != "H"]
        if not moving_idx:
            raise ValueError("no movable atoms (all fixed?)")
        alts = {mol.keys[i][4] for i in moving_idx} - {""}
        self.altloc = altloc or (sorted(alts)[0] if alts else "")
        moving_idx = [i for i in moving_idx if mol.keys[i][4] in ("", self.altloc)]
        self._build(moving_idx, cif_paths, shell)

    # ------------------------------------------------------------------ setup
    def _build(self, moving_idx, cif_paths, shell):
        mol = self.mol
        moving_res = {mol.keys[i][:3] for i in moving_idx}
        # include sequence neighbours so peptide links to the fixed world exist
        seq_nb = set()
        for ch, num, ic in moving_res:
            seq_nb |= {(ch, num - 1, ""), (ch, num + 1, "")}
        keep_res = moving_res | seq_nb | mol.neighbours_of_residues(moving_res, shell)

        sub = mol.st.clone()
        sub.remove_hydrogens()
        for ch in sub[0]:
            for ri in range(len(ch) - 1, -1, -1):
                res = ch[ri]
                if (ch.name, res.seqid.num, res.seqid.icode.strip()) not in keep_res:
                    del ch[ri]
                    continue
                # TorchRef only links ATOM records: treat modified amino acids (KCX, ...) as polymer
                if res.het_flag == "H" and res.find_atom("CA", "*") and res.find_atom("N", "*") and res.find_atom("C", "*"):
                    res.het_flag = "A"
                for ai in range(len(res) - 1, -1, -1):
                    a = res[ai]
                    al = a.altloc if a.altloc != "\0" else ""
                    if al not in ("", self.altloc):
                        del res[ai]
                    else:
                        a.altloc = "\0"
                        a.occ = 1.0
        sub.remove_empty_chains()
        sub.setup_entities()
        fd, tmp = tempfile.mkstemp(suffix=".pdb", prefix="moorlight_rsr_")
        os.close(fd)
        sub.write_pdb(tmp)

        t0 = time.time()
        try:
            rr, atoms = load_restraints(tmp, cif_paths)
        finally:
            os.unlink(tmp)
        self.build_time = time.time() - t0

        # map sub-model atoms -> molecule atoms
        lut = {}
        for i, k in enumerate(mol.keys):
            if k[4] in ("", self.altloc):
                lut[k[:4]] = i
        sub_to_mol = np.array([lut.get(a, -1) for a in atoms])
        if (sub_to_mol < 0).any():
            missing = [atoms[i] for i in np.nonzero(sub_to_mol < 0)[0][:5]]
            raise RuntimeError(f"could not map sub-model atoms back: {missing}")
        self.sub_to_mol = sub_to_mol
        mol_to_sub = {int(j): i for i, j in enumerate(sub_to_mol)}
        moving_sub = sorted(mol_to_sub[i] for i in moving_idx if i in mol_to_sub)
        self.moving_sub = torch.tensor(moving_sub, dtype=torch.long)
        self.moving_mol = sub_to_mol[moving_sub]
        self.mol_to_moving = {int(j): k for k, j in enumerate(self.moving_mol)}

        xyz0 = torch.tensor(mol.xyz[sub_to_mol], dtype=torch.float64)
        self.xyz_fixed = xyz0.clone()
        self.x = torch.nn.Parameter(xyz0[self.moving_sub].clone())
        self.start_xyz = self.x.detach().clone()
        is_moving = torch.zeros(len(xyz0), dtype=torch.bool)
        is_moving[self.moving_sub] = True

        def keep(idx):
            """restraints touching at least one moving atom and no bogus (gap) pairs"""
            return is_moving[idx].any(dim=1)

        self.terms = {}
        b = rr["bond"]["all"]
        bi, bref, bsig = b["indices"].long(), b["references"].double(), b["sigmas"].double()
        # drop bonds that are clearly not bonds (chain gaps inside the sub-model)
        d0 = (xyz0[bi[:, 0]] - xyz0[bi[:, 1]]).norm(dim=1)
        bad = (d0 - bref).abs() > 1.2
        bad_pairs = {tuple(sorted(p)) for p in bi[bad].tolist()}

        def drop_bad(idx):
            if not bad_pairs:
                return torch.ones(len(idx), dtype=torch.bool)
            ok = []
            for row in idx.tolist():
                s = set(row)
                ok.append(not any(a in s and c in s for a, c in bad_pairs))
            return torch.tensor(ok, dtype=torch.bool)

        sel = keep(bi) & ~bad
        self.terms["bond"] = (bi[sel], bref[sel], bsig[sel])
        a = rr["angle"]["all"]
        ai = a["indices"].long()
        sel = keep(ai) & drop_bad(ai)
        self.terms["angle"] = (ai[sel], a["references"].double()[sel] * DEG, a["sigmas"].double()[sel] * DEG)
        t = rr["torsion"]["all"]
        if t["indices"] is not None and len(t["indices"]):
            ti = t["indices"].long()
            tref, tsig, tper = t["references"].double(), t["sigmas"].double(), t["periods"].long()
            # dictionaries (e.g. grade2) contain torsions with period 0 or sigma 0: unusable, would give NaN
            sel = keep(ti) & drop_bad(ti) & (tper > 0) & (tsig > 0) & torch.isfinite(tref)
            self.terms["torsion"] = (ti[sel], tref[sel], tsig[sel], tper[sel])
        if "omega" in rr["torsion"].keys():
            oi = rr["torsion"]["omega"]["indices"].long()
            sel = keep(oi) & drop_bad(oi)
            oi = oi[sel]
            # keep current cis/trans state
            cur = _dihedral(xyz0, oi)
            ref = torch.where(cur.abs() < 90 * DEG, torch.zeros_like(cur), torch.full_like(cur, np.pi))
            self.terms["omega"] = (oi, ref)
        planes = []
        for key in rr["plane"].keys():
            pi = rr["plane"][key]["indices"]
            if pi is None or pi.shape[1] <= 3:
                continue
            pi = pi.long()
            sel = keep(pi) & drop_bad(pi)
            if sel.any():
                planes.append((pi[sel], rr["plane"][key]["sigmas"].double()[sel]))
        self.terms["plane"] = planes
        if "chiral" in rr and rr["chiral"].get("indices") is not None:
            ci = rr["chiral"]["indices"].long()
            sel = keep(ci) & drop_bad(ci)
            self.terms["chiral"] = (ci[sel], rr["chiral"]["ideal_volumes"].double()[sel],
                                    rr["chiral"]["sigmas"].double()[sel])
        # exclusions for non-bonded: 1-2 and 1-3 pairs
        excl = set()
        for row in bi[~bad].tolist():
            excl.add(tuple(sorted(row)))
        for row in ai.tolist():
            excl.add(tuple(sorted((row[0], row[2]))))
        self.excl = excl
        self.elements_sub = [mol.elements[j] for j in sub_to_mol]
        self.nb = None
        self.diff = None
        self._update_nonbonded()
        self._setup_density()
        self._setup_difference()

    def _setup_density(self):
        if self.xmap is None or self.weight <= 0:
            self.dens = None
            return
        x = self.start_xyz.numpy()
        lo, hi = x.min(0) - 6.0, x.max(0) + 6.0
        center = (lo + hi) / 2
        radius = float((hi - lo).max() / 2)
        arr, origin, step = self.xmap.box(center, radius)
        arr = (arr - self.xmap.mean) / self.xmap.rms
        # grid_sample wants (N, C, D, H, W) with grid coords ordered (x=W, y=H, z=D)
        self.dens = torch.tensor(arr, dtype=torch.float64).permute(2, 1, 0)[None, None].contiguous()
        self.dens_origin = torch.tensor(origin, dtype=torch.float64)
        self.dens_extent = torch.tensor([(s - 1) * step for s in arr.shape], dtype=torch.float64)
        zw = {"S": 1.6, "P": 1.5, "SE": 2.0}
        self.dens_w = torch.tensor([zw.get(self.elements_sub[i].upper(), 1.0) for i in self.moving_sub.tolist()],
                                   dtype=torch.float64)

    def _density(self, x):
        g = (x - self.dens_origin) / self.dens_extent * 2 - 1
        v = torch.nn.functional.grid_sample(self.dens, g[None, None, None], mode="bilinear",
                                            padding_mode="border", align_corners=True)
        return v.view(-1)

    # ------------------------------------------------------- difference target
    def _dark_reference(self):
        """Dark counterpart atoms of the moving (light) atoms: list of (xyz, element, B, weight)."""
        mol, out = self.mol, []
        dark = self.dark_mol if self.dark_mol is not None and self.dark_mol is not mol else None
        if dark is not None:
            by_name = {}
            for i, k in enumerate(dark.keys):
                by_name.setdefault(k[:4], []).append(i)
        else:
            by_name = {}
            for i, k in enumerate(mol.keys):
                if k[4] not in ("", self.altloc):
                    by_name.setdefault(k[:4], []).append(i)
        for k, mi in enumerate(self.moving_mol):
            key = mol.keys[mi][:4]
            cands = by_name.get(key, [])
            src = dark if dark is not None else mol
            if cands:
                occ = np.array([src.info[c][6] for c in cands], float)
                w = occ / occ.sum() if occ.sum() > 0 else np.full(len(cands), 1 / len(cands))
                for c, wc in zip(cands, w):
                    out.append((src.xyz[c], src.elements[c], src.info[c][7], wc))
            elif dark is None:
                # no other conformer: the dark state is where the atom started
                out.append((self.start_xyz[k].numpy(), mol.elements[mi], mol.info[mi][7], 1.0))
        return out

    def _setup_difference(self):
        self.diff = None
        if self.diff_map is None or self.diff_weight <= 0:
            return
        ref = self._dark_reference()
        mol = self.mol
        zmap = {"H": 1, "C": 6, "N": 7, "O": 8, "S": 16, "P": 15, "SE": 34, "MG": 12, "FE": 26, "ZN": 30, "CL": 17}
        bres = 4.0 * getattr(self.diff_map, "resolution", 2.0) ** 2
        lz = [zmap.get(mol.elements[i].upper(), 6) for i in self.moving_mol]
        lb = [max(mol.info[i][7], 5.0) + bres for i in self.moving_mol]
        self.diff = {
            "lz": torch.tensor(lz, dtype=torch.float64), "lb": torch.tensor(lb, dtype=torch.float64),
            "dx": torch.tensor(np.array([r[0] for r in ref]).reshape(-1, 3), dtype=torch.float64),
            "dz": torch.tensor([zmap.get(r[1].upper(), 6) for r in ref], dtype=torch.float64),
            "db": torch.tensor([max(r[2], 5.0) + bres for r in ref], dtype=torch.float64),
            "dw": torch.tensor([r[3] for r in ref], dtype=torch.float64),
        }
        pts = np.concatenate([self.start_xyz.numpy(), self.diff["dx"].numpy()])
        lo, hi = pts.min(0) - 5.0, pts.max(0) + 5.0
        arr, origin, step = self.diff_map.box((lo + hi) / 2, float((hi - lo).max() / 2))
        arr = (arr - self.diff_map.mean) / self.diff_map.rms
        n = arr.shape[0]
        g = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1).reshape(-1, 3) * step + origin
        self.diff.update(grid=g, obs_all=arr.reshape(-1), step=step)
        self._update_diff_mask()
        # scale: fit to the starting model if it already differs from the dark reference, otherwise
        # assume a fully moved carbon would give a ~4 sigma peak
        with torch.no_grad():
            calc = self._diff_calc(self.x, unit=True)
            obs = self.diff["obs"]
            cc = float((calc * calc).sum())
            peak_c = 6 * (4 * np.pi / (20 + bres)) ** 1.5
            k0 = float((obs * calc).sum() / cc) if cc > 1e-6 * len(calc) * peak_c ** 2 else 0.0
            if k0 <= 0:
                k0 = 4.0 / peak_c
        self.diff["logk0"] = float(np.log(k0))
        self.logk = torch.nn.Parameter(torch.tensor(np.log(k0), dtype=torch.float64))

    def _update_diff_mask(self, radius=3.0):
        d = self.diff
        cur = self.x.detach().numpy()
        pts = np.concatenate([self.start_xyz.numpy(), cur, d["dx"].numpy()])
        dist, _ = cKDTree(pts).query(d["grid"], distance_upper_bound=radius)
        m = np.isfinite(dist)
        d["pts"] = torch.tensor(d["grid"][m], dtype=torch.float64)
        d["obs"] = torch.tensor(d["obs_all"][m], dtype=torch.float64)

    def _diff_calc(self, x, unit=False):
        """Model difference density (light - dark) at the masked grid points; unit=True: without scale k."""
        d = self.diff
        xyz = torch.cat([x, d["dx"]])
        z = torch.cat([d["lz"], -d["dz"] * d["dw"]])
        b = torch.cat([d["lb"], d["db"]])
        r2 = torch.cdist(d["pts"], xyz) ** 2
        g = (4 * np.pi / b) ** 1.5 * torch.exp(-4 * np.pi ** 2 * r2 / b)
        calc = g @ z
        return calc if unit else calc * torch.exp(self.logk)

    def diff_stats(self):
        if self.diff is None:
            return {}
        with torch.no_grad():
            calc = self._diff_calc(self.x)
            obs = self.diff["obs"]
            c = calc - calc.mean()
            o = obs - obs.mean()
            cc = float((c * o).sum() / (c.norm() * o.norm() + 1e-12))
        return {"diff_cc": cc, "diff_scale": float(torch.exp(self.logk))}

    def _update_nonbonded(self):
        if self.diff is not None:
            self._update_diff_mask()
        if not self.use_nonbonded:
            self.nb = None
            return
        xyz = self.full_xyz().detach().numpy()
        tree = cKDTree(xyz)
        mov = self.moving_sub.numpy()
        pairs = set()
        for i, nb in zip(mov, tree.query_ball_point(xyz[mov], 4.2)):
            for j in nb:
                if i == j:
                    continue
                p = (min(i, j), max(i, j))
                if p not in self.excl:
                    pairs.add(p)
        if pairs:
            p = torch.tensor(sorted(pairs), dtype=torch.long)
            el = self.elements_sub
            polar = torch.tensor([el[k] in ("N", "O") for k in range(len(el))])
            hb = polar[p[:, 0]] & polar[p[:, 1]]
            dmin = torch.where(hb, torch.full((len(p),), 2.55, dtype=torch.float64),
                               torch.full((len(p),), 3.0, dtype=torch.float64))
            self.nb = (p, dmin)
        else:
            self.nb = None

    # --------------------------------------------------------------- energy
    def full_xyz(self):
        xyz = self.xyz_fixed.clone()
        xyz[self.moving_sub] = self.x
        return xyz

    def energy(self, parts=False):
        xyz = self.full_xyz()
        e = {}
        bi, bref, bsig = self.terms["bond"]
        if len(bi):
            d = (xyz[bi[:, 0]] - xyz[bi[:, 1]]).norm(dim=1)
            e["bond"] = (0.5 * ((d - bref) / bsig) ** 2).sum()
        ai, aref, asig = self.terms["angle"]
        if len(ai):
            v1 = xyz[ai[:, 0]] - xyz[ai[:, 1]]
            v2 = xyz[ai[:, 2]] - xyz[ai[:, 1]]
            cos = (v1 * v2).sum(1) / (v1.norm(dim=1) * v2.norm(dim=1))
            ang = torch.acos(cos.clamp(-0.999999, 0.999999))
            e["angle"] = (0.5 * ((ang - aref) / asig) ** 2).sum()
        if self.use_torsions and "torsion" in self.terms and len(self.terms["torsion"][0]):
            ti, tref, tsig, per = self.terms["torsion"]
            ang = _dihedral(xyz, ti)
            dev = ang - tref * DEG
            dev = torch.atan2(torch.sin(per * dev), torch.cos(per * dev)) / per
            e["torsion"] = 0.2 * (0.5 * (dev / (tsig * DEG)) ** 2).sum()
        if "omega" in self.terms and len(self.terms["omega"][0]):
            oi, oref = self.terms["omega"]
            ang = _dihedral(xyz, oi)
            dev = torch.atan2(torch.sin(ang - oref), torch.cos(ang - oref))
            e["omega"] = (0.5 * (dev / (5 * DEG)) ** 2).sum()
        if self.terms["plane"]:
            tot = 0
            for pi, ps in self.terms["plane"]:
                pos = xyz[pi]
                c = pos - pos.mean(dim=1, keepdim=True)
                with torch.no_grad():
                    n = torch.linalg.svd(c.detach(), full_matrices=False)[2][:, -1, :]
                dev = torch.einsum("paj,pj->pa", c, n)
                tot = tot + (0.5 * (dev / ps) ** 2).sum()
            e["plane"] = tot
        if "chiral" in self.terms and len(self.terms["chiral"][0]):
            ci, iv, cs = self.terms["chiral"]
            pc = xyz[ci[:, 0]]
            vol = ((xyz[ci[:, 1]] - pc) * torch.cross(xyz[ci[:, 2]] - pc, xyz[ci[:, 3]] - pc, dim=-1)).sum(1)
            vol = torch.where(iv == 0, vol.abs(), vol)
            ideal = torch.where(iv == 0, torch.full_like(iv, 2.5), iv)
            e["chiral"] = (0.5 * ((vol - ideal) / cs) ** 2).sum()
        if self.nb is not None:
            p, dmin = self.nb
            d = (xyz[p[:, 0]] - xyz[p[:, 1]]).norm(dim=1)
            e["nonbonded"] = (0.5 * (torch.clamp(dmin - d, min=0) / 0.012) ** 2).sum()
        if self.dens is not None and self.weight > 0:
            rho = self._density(self.x)
            e["density"] = -self.weight * (self.dens_w * rho).sum()
        if self.diff is not None:
            res = self.diff["obs"] - self._diff_calc(self.x)
            # 0.05 puts weight 60 on a similar footing as the map term
            e["difference"] = 0.05 * self.diff_weight * 0.5 * (res ** 2).sum() * self.diff["step"] ** 3
            e["diff_scale_prior"] = 0.5 * (self.logk - self.diff["logk0"]) ** 2
        with self.lock:
            pulls = dict(self.pulls)
        if pulls:
            idx = torch.tensor([self.mol_to_moving[i] for i in pulls], dtype=torch.long)
            tgt = torch.tensor(np.array(list(pulls.values())), dtype=torch.float64)
            # harmonic near the mouse, constant force (pull_k * a) far away, so a strong pull cannot
            # override non-bonded repulsion when that is switched on
            a = 0.6
            r2 = ((self.x[idx] - tgt) ** 2).sum(1)
            e["pull"] = (self.pull_k * a * a * (torch.sqrt(1 + r2 / (a * a)) - 1)).sum()
        total = sum(e.values())
        return (total, e) if parts else total

    def geometry_stats(self):
        with torch.no_grad():
            xyz = self.full_xyz()
            bi, bref, bsig = self.terms["bond"]
            ai, aref, asig = self.terms["angle"]
            out = {}
            if len(bi):
                d = (xyz[bi[:, 0]] - xyz[bi[:, 1]]).norm(dim=1)
                out["bond_rmsd"] = float(((d - bref) ** 2).mean().sqrt())
                out["bond_max_z"] = float(((d - bref) / bsig).abs().max())
            if len(ai):
                v1 = xyz[ai[:, 0]] - xyz[ai[:, 1]]
                v2 = xyz[ai[:, 2]] - xyz[ai[:, 1]]
                ang = torch.acos(((v1 * v2).sum(1) / (v1.norm(dim=1) * v2.norm(dim=1))).clamp(-1, 1))
                out["angle_rmsd_deg"] = float(((ang - aref) ** 2).mean().sqrt() / DEG)
            if self.dens is not None and self.weight > 0:
                out["mean_density_sigma"] = float(self._density(self.x).mean())
            out["max_shift"] = float((self.x - self.start_xyz).norm(dim=1).max())
        out.update(self.diff_stats())
        return out

    # ------------------------------------------------------------- running
    def step(self, n_iter=20):
        params = [self.x] + ([self.logk] if self.diff is not None else [])
        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=n_iter, history_size=20,
                                line_search_fn="strong_wolfe", tolerance_change=1e-7)

        def closure():
            opt.zero_grad()
            loss = self.energy()
            loss.backward()
            return loss

        loss = opt.step(closure)
        return float(loss.detach())

    def run(self, max_steps=200, until_converged=True):
        """Blocking refinement loop; returns when converged (and not pulled) or stopped."""
        self.running = True
        last, it = None, 0
        try:
            while self.running and it < max_steps:
                it += 1
                if it % 5 == 0:
                    self._update_nonbonded()
                loss = self.step()
                pulled = bool(self.pulls)
                if self.on_update:
                    self.on_update(self)
                if until_converged and not pulled and last is not None and abs(last - loss) < 1e-3 * max(1, abs(loss)) * 0.01:
                    break
                last = loss
                if pulled:
                    time.sleep(0.01)
        finally:
            self.running = False
        tot, parts = self.energy(parts=True)
        self.status = {"energy": float(tot), "terms": {k: float(v) for k, v in parts.items()},
                       **self.geometry_stats(), "steps": it}
        return self.status

    def moving_coords(self):
        return self.x.detach().numpy().copy()

    def set_pull(self, mol_idx, target):
        if mol_idx not in self.mol_to_moving:
            return False
        with self.lock:
            if target is None:
                self.pulls.pop(mol_idx, None)
            else:
                self.pulls[mol_idx] = np.asarray(target, dtype=float)
        return True

    def apply(self):
        """Write refined coordinates into the molecule (caller checkpoints first)."""
        mol = self.mol
        new = self.moving_coords()
        shift = {}
        for k, mi in enumerate(self.moving_mol):
            shift[mol.keys[mi]] = new[k] - mol.xyz[mi]
        model = mol.st[0]
        for ch in model:
            for res in ch:
                heavy = {}
                for at in res:
                    key = mol.atom_key(ch, res, at)
                    if key in shift:
                        s = shift[key]
                        at.pos = gemmi.Position(*(np.array(at.pos.tolist()) + s))
                        heavy[at.name] = s
                if heavy:  # drag riding hydrogens along with their nearest moved heavy atom
                    for at in res:
                        if at.element.name == "H":
                            al = at.altloc if at.altloc != "\0" else ""
                            if al not in ("", self.altloc):
                                continue
                            best = min(((at.pos.dist(b.pos), b.name) for b in res
                                        if b.element.name != "H" and b.name in heavy), default=None)
                            if best and best[0] < 1.3:
                                at.pos = gemmi.Position(*(np.array(at.pos.tolist()) + heavy[best[1]]))
        mol.rebuild()


def _dihedral(xyz, idx):
    p0, p1, p2, p3 = (xyz[idx[:, k]] for k in range(4))
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = b1 / b1.norm(dim=1, keepdim=True)
    v = b0 - (b0 * b1n).sum(1, keepdim=True) * b1n
    w = b2 - (b2 * b1n).sum(1, keepdim=True) * b1n
    x = (v * w).sum(1)
    y = (torch.cross(b1n, v, dim=-1) * w).sum(1)
    return torch.atan2(y, x)


def warm_up():
    """Import TorchRef and build restraints for a dipeptide so the first real refinement is fast."""
    st = gemmi.Structure()
    st.cell = gemmi.UnitCell(30, 30, 30, 90, 90, 90)
    st.spacegroup_hm = "P 1"
    st.add_model(gemmi.Model("1"))
    ch = st[0].add_chain("A")
    for num, name, xyz in ((1, "SER", [(0, 0, 0), (1.46, 0, 0), (2.0, 1.42, 0), (1.25, 2.39, 0), (2.0, -0.8, 1.2), (3.4, -0.8, 1.2)]),
                           (2, "GLY", [(3.33, 1.55, 0), (3.95, 2.86, 0), (5.46, 2.75, 0), (6.1, 3.8, 0)])):
        r = gemmi.Residue()
        r.name, r.seqid = name, gemmi.SeqId(num, " ")
        for an, p in zip(("N", "CA", "C", "O", "CB", "OG"), xyz):
            a = gemmi.Atom()
            a.name, a.element, a.pos, a.occ, a.b_iso = an, gemmi.Element(an[0]), gemmi.Position(*p), 1.0, 20.0
            r.add_atom(a)
        ch.add_residue(r)
    st.setup_entities()
    fd, tmp = tempfile.mkstemp(suffix=".pdb")
    os.close(fd)
    st.write_pdb(tmp)
    try:
        load_restraints(tmp)
    finally:
        os.unlink(tmp)
