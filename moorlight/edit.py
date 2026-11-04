"""Model building operations on a Molecule (gemmi Structure)."""
from __future__ import annotations

import os

import gemmi
import numpy as np
from scipy.spatial import cKDTree

from .model import Molecule, WATER

MAIN_CHAIN = {"N", "CA", "C", "O", "OXT", "H", "HA", "H1", "H2", "H3"}

# ---------------------------------------------------------------- rotamers
CHI_ATOMS = {
    "SER": [("N", "CA", "CB", "OG")],
    "THR": [("N", "CA", "CB", "OG1")],
    "CYS": [("N", "CA", "CB", "SG")],
    "VAL": [("N", "CA", "CB", "CG1")],
    "LEU": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "ILE": [("N", "CA", "CB", "CG1"), ("CA", "CB", "CG1", "CD1")],
    "PHE": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "TYR": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "TRP": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "HIS": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "ND1")],
    "ASN": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")],
    "ASP": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")],
    "MET": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "SD"), ("CB", "CG", "SD", "CE")],
    "GLU": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "OE1")],
    "GLN": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "OE1")],
    "LYS": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "CE"), ("CG", "CD", "CE", "NZ")],
    "ARG": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "NE"), ("CG", "CD", "NE", "CZ")],
}
_P, _T, _M = 62.0, -177.0, -65.0
# Penultimate rotamer library (Lovell et al. 2000), modal chi values.
ROTAMERS = {
    "SER": {"p": [_P], "t": [_T], "m": [_M]},
    "THR": {"p": [59], "t": [-171], "m": [-61]},
    "CYS": {"p": [_P], "t": [_T], "m": [_M]},
    "VAL": {"p": [63], "t": [175], "m": [-60]},
    "LEU": {"pp": [62, 80], "tp": [-177, 65], "tt": [-172, 145], "mp": [-85, 65], "mt": [-65, 175]},
    "ILE": {"pp": [62, 100], "pt": [62, 170], "tp": [-177, 66], "tt": [-177, 165], "mp": [-65, 100],
            "mt": [-65, 170], "mm": [-57, -60]},
    "PHE": {"p90": [62, 90], "t80": [-177, 80], "m-85": [-65, -85], "m-30": [-65, -30]},
    "TYR": {"p90": [62, 90], "t80": [-177, 80], "m-85": [-65, -85], "m-30": [-65, -30]},
    "TRP": {"p-90": [62, -90], "p90": [62, 90], "t-105": [-177, -105], "t90": [-177, 90],
            "m-90": [-65, -90], "m0": [-65, -5], "m95": [-65, 95]},
    "HIS": {"p-80": [62, -75], "p80": [62, 80], "t-160": [-177, -165], "t-80": [-177, -80],
            "t60": [-177, 60], "m-70": [-65, -70], "m170": [-65, 165], "m80": [-65, 80]},
    "ASN": {"p-10": [62, -10], "p30": [62, 30], "t-20": [-174, -20], "t30": [-177, 30],
            "m-20": [-65, -20], "m-80": [-65, -75], "m120": [-65, 120]},
    "ASP": {"p-10": [62, -10], "p30": [62, 30], "t0": [-177, 0], "t70": [-177, 65], "m-20": [-70, -15]},
    "MET": {"ptp": [62, 180, 75], "ptm": [62, 180, -75], "tpp": [-177, 65, 75], "tpt": [-177, 65, 180],
            "ttp": [-177, 180, 75], "ttt": [-177, 180, 180], "ttm": [-177, 180, -75], "mtp": [-65, 180, 75],
            "mtt": [-65, 180, 180], "mtm": [-65, 180, -75], "mmp": [-65, -65, 103], "mmt": [-65, -65, 180],
            "mmm": [-65, -65, -70]},
    "GLU": {"pt-20": [62, 180, -20], "pm0": [70, -80, 0], "tp10": [-177, 65, 10], "tt0": [-177, 180, 0],
            "tm-20": [-177, -80, -25], "mp0": [-65, 85, 0], "mt-10": [-67, 180, -10], "mm-40": [-65, -65, -40]},
    "GLN": {"pt20": [62, 180, 20], "pm0": [70, -75, 0], "tp-100": [-177, 65, -100], "tp60": [-177, 65, 60],
            "tt0": [-177, 180, 0], "mp0": [-65, 85, 0], "mt-30": [-67, 180, -25], "mm-40": [-65, -65, -40],
            "mm100": [-65, -65, 100]},
}
_L = {"p": 62.0, "t": 180.0, "m": -65.0}
ROTAMERS["LYS"] = {n: [_L[n[0]] if n[0] != "t" else -177.0, _L[n[1]], _L[n[2]], _L[n[3]] if n[3] != "p" else 65.0]
                   for n in ["ptpt", "pttp", "pttt", "pttm", "ptmt", "tptp", "tptt", "tptm", "ttpp", "ttpt",
                             "tttp", "tttt", "tttm", "ttmt", "ttmm", "mptt", "mttp", "mttt", "mttm", "mtmt",
                             "mtmm", "mmtp", "mmtt", "mmtm", "mmmt"]}
ROTAMERS["ARG"] = {}
for _n in ["ptp85", "ptp180", "ptt85", "ptt180", "ptt-85", "ptm180", "ptm-85", "tpp80", "tpt85", "tpt170",
           "ttp85", "ttp180", "ttp-105", "ttt85", "ttt180", "ttt-85", "ttm105", "ttm180", "ttm-85",
           "mtp85", "mtp180", "mtp-105", "mtt85", "mtt180", "mtt-85", "mtm105", "mtm180", "mtm-85",
           "mmt85", "mmt180", "mmt-85", "mmm180", "mmm-85"]:
    ROTAMERS["ARG"][_n] = [-177.0 if _n[0] == "t" else _L[_n[0]], _L[_n[1]], _L[_n[2]], float(_n[3:])]


def _dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - b0.dot(b1) * b1
    w = b2 - b2.dot(b1) * b1
    return np.degrees(np.arctan2(np.cross(b1, v).dot(w), v.dot(w)))


def _rotate(points, origin, axis, angle_deg):
    axis = axis / np.linalg.norm(axis)
    a = np.radians(angle_deg)
    p = points - origin
    rot = (p * np.cos(a) + np.cross(axis, p) * np.sin(a) + np.outer(p @ axis, axis) * (1 - np.cos(a)))
    return rot + origin


def _res_atoms(res, altloc=""):
    """name -> atom for blank + given altloc (first conformer if altloc empty)."""
    out = {}
    for at in res:
        al = at.altloc if at.altloc != "\0" else ""
        if al in ("", altloc) or (not altloc and at.name not in out):
            out.setdefault(at.name, at)
    return out


def _downstream(res_atoms, a2, a3):
    """Atoms reachable from a3 without passing a2 (intra-residue bond graph)."""
    names = [n for n, a in res_atoms.items() if a.element.name != "H" or True]
    pos = {n: np.array(res_atoms[n].pos.tolist()) for n in names}
    adj = {n: set() for n in names}
    for i, n in enumerate(names):
        for m in names[i + 1:]:
            if np.linalg.norm(pos[n] - pos[m]) < 1.95:
                adj[n].add(m)
                adj[m].add(n)
    seen, todo = {a3}, [a3]
    while todo:
        cur = todo.pop()
        for nb in adj[cur]:
            if nb not in seen and nb != a2 and nb not in ("N", "C", "CA"):
                seen.add(nb)
                todo.append(nb)
    seen.discard(a3)
    return seen


def get_chis(res, altloc=""):
    ats = _res_atoms(res, altloc)
    out = []
    for quad in CHI_ATOMS.get(res.name, []):
        if all(q in ats for q in quad):
            out.append(_dihedral(*[np.array(ats[q].pos.tolist()) for q in quad]))
    return out


def set_chis(res, chis, altloc=""):
    ats = _res_atoms(res, altloc)
    for k, (quad, target) in enumerate(zip(CHI_ATOMS[res.name], chis)):
        if not all(q in ats for q in quad):
            raise ValueError(f"residue lacks atoms for chi{k + 1}")
        p = [np.array(ats[q].pos.tolist()) for q in quad]
        delta = target - _dihedral(*p)
        move = _downstream(ats, quad[1], quad[2])
        for n in move:
            new = _rotate(np.array([ats[n].pos.tolist()]), p[2], p[2] - p[1], delta)[0]
            ats[n].pos = gemmi.Position(*new)


def rotamer_list(mol: Molecule, chain, num, icode="", altloc="", xmap=None):
    """All library rotamers for a residue, scored by density (sigma) when a map is given."""
    ch, ri, res = mol.find_residue(chain, num, icode)
    if res.name not in ROTAMERS:
        return []
    orig = {(a.name, a.altloc): a.pos.tolist() for a in res}
    out = []
    current = get_chis(res, altloc)
    for name, chis in ROTAMERS[res.name].items():
        set_chis(res, chis, altloc)
        score = None
        if xmap is not None:
            ats = _res_atoms(res, altloc)
            vals = [xmap.value_at(a.pos.tolist()) for n, a in ats.items() if n not in MAIN_CHAIN and a.element.name != "H"]
            score = float((np.mean(vals) - xmap.mean) / xmap.rms) if vals else 0.0
        out.append({"name": name, "chis": chis, "density": score})
        for a in res:
            a.pos = gemmi.Position(*orig[(a.name, a.altloc)])
    if xmap is not None:
        out.sort(key=lambda r: -r["density"])
    return {"current_chis": current, "rotamers": out}


def apply_rotamer(mol, chain, num, name, icode="", altloc=""):
    ch, ri, res = mol.find_residue(chain, num, icode)
    set_chis(res, ROTAMERS[res.name][name], altloc)
    mol.rebuild()


# ------------------------------------------------------------- alt confs
def add_altconf(mol, chain, num, icode="", whole_residue=False, occ_new=0.5):
    """Split a residue: side chain (+CA) or whole residue gets a new alternative conformer.
    Returns the new altloc letter."""
    ch, ri, res = mol.find_residue(chain, num, icode)
    existing = sorted({a.altloc for a in res if a.altloc not in ("\0", " ")})
    split_names = {a.name for a in res if whole_residue or a.name not in ("N", "C", "O", "H")}
    if not existing:
        new = "B"
        add = []
        for a in res:
            if a.name in split_names:
                a.altloc = "A"
                a.occ = 1.0 - occ_new
                b = a.clone()
                b.altloc = "B"
                b.occ = occ_new
                add.append(b)
        for b in add:
            res.add_atom(b)
    else:
        new = chr(ord(existing[-1]) + 1)
        src = existing[0]
        n = len(existing)
        add = []
        for a in res:
            if a.altloc in existing:
                a.occ = a.occ * (1 - occ_new)
            if a.altloc == src:
                b = a.clone()
                b.altloc = new
                b.occ = occ_new
                add.append(b)
        for b in add:
            res.add_atom(b)
    _sort_residue_atoms(res)
    mol.rebuild()
    return new


def _sort_residue_atoms(res):
    atoms = [a.clone() for a in res]
    order = {}
    for a in atoms:
        order.setdefault(a.name, len(order))
    atoms.sort(key=lambda a: (order[a.name], a.altloc))
    for i in range(len(res) - 1, -1, -1):
        del res[i]
    for a in atoms:
        res.add_atom(a)


def remove_altconf(mol, chain, num, altloc, icode=""):
    ch, ri, res = mol.find_residue(chain, num, icode)
    for i in range(len(res) - 1, -1, -1):
        if res[i].altloc == altloc:
            del res[i]
    left = sorted({a.altloc for a in res if a.altloc not in ("\0", " ")})
    if len(left) == 1:
        for a in res:
            a.altloc = "\0"
            a.occ = 1.0
    mol.rebuild()


# ------------------------------------------------------------- monomers
def monomer_library_dir():
    import torchref
    return os.path.join(os.path.dirname(torchref.__file__), "data", "monomer_library")


def ideal_residue(resname, cif_path=None):
    """Residue with ideal coordinates from a restraint CIF (default: TorchRef's monomer library)."""
    if cif_path is None:
        cif_path = os.path.join(monomer_library_dir(), resname[0].lower(), resname.upper() + ".cif")
    doc = gemmi.cif.read(cif_path)
    block = doc.find_block("comp_" + resname) or next(
        (b for b in doc if b.name.startswith("comp_") and b.name != "comp_list"), None)
    if block is None:
        raise ValueError(f"no comp_ block for {resname} in {cif_path}")
    res = residue_from_comp_block(block)
    return res, block.name[5:]


def residue_from_comp_block(block):
    """Build a gemmi.Residue from _chem_comp_atom coordinates (x/y/z, model_Cartn or ideal columns)."""
    tab = None
    for cols in (["x", "y", "z"], ["model_Cartn_x", "model_Cartn_y", "model_Cartn_z"],
                 ["pdbx_model_Cartn_x_ideal", "pdbx_model_Cartn_y_ideal", "pdbx_model_Cartn_z_ideal"]):
        t = block.find("_chem_comp_atom.", ["atom_id", "type_symbol"] + cols)
        if len(t):
            vals = [[gemmi.cif.as_number(r[k]) for k in (2, 3, 4)] for r in t]
            if all(all(v == v for v in xyz) for xyz in vals) and any(any(v != 0 for v in xyz) for xyz in vals):
                tab = (t, vals)
                break
    if tab is None:
        raise ValueError(f"{block.name}: dictionary has no coordinates (generate them with acedrg/grade/eLBOW)")
    res = gemmi.Residue()
    res.name = block.name[5:]
    for row, xyz in zip(*tab):
        a = gemmi.Atom()
        a.name = gemmi.cif.as_string(row[0])
        a.element = gemmi.Element(gemmi.cif.as_string(row[1]))
        a.pos = gemmi.Position(*xyz)
        a.occ = 1.0
        res.add_atom(a)
    return res


def _superpose(mobile_pts, target_pts):
    """Kabsch: returns (R, t) mapping mobile onto target."""
    mc, tc = mobile_pts.mean(0), target_pts.mean(0)
    h = (mobile_pts - mc).T @ (target_pts - tc)
    u, s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    return r, tc - r @ mc


def mutate(mol, chain, num, new_resname, icode=""):
    ch, ri, res = mol.find_residue(chain, num, icode)
    new_resname = new_resname.upper()
    ideal, _ = ideal_residue(new_resname)
    ideal_ats = {a.name: a for a in ideal if a.element.name != "H" and a.name != "OXT"}
    have = _res_atoms(res)
    names = [n for n in ("N", "CA", "C") if n in have and n in ideal_ats]
    if len(names) < 3:
        raise ValueError("need N, CA, C to mutate")
    r, t = _superpose(np.array([ideal_ats[n].pos.tolist() for n in names]),
                      np.array([have[n].pos.tolist() for n in names]))
    old_chis = get_chis(res)
    template = next(iter(res))
    for i in range(len(res) - 1, -1, -1):
        if res[i].name not in MAIN_CHAIN or res[i].element.name == "H":
            del res[i]
    for n, a in ideal_ats.items():
        if n in MAIN_CHAIN:
            continue
        b = gemmi.Atom()
        b.name, b.element = n, a.element
        b.pos = gemmi.Position(*(r @ np.array(a.pos.tolist()) + t))
        b.occ, b.b_iso = template.occ, template.b_iso
        b.altloc = template.altloc
        res.add_atom(b)
    res.name = new_resname
    _sort_residue_atoms(res)
    # keep as many of the old chi angles as make sense
    if new_resname in CHI_ATOMS and old_chis:
        n = min(len(old_chis), len(CHI_ATOMS[new_resname]))
        try:
            set_chis(res, old_chis[:n] + get_chis(res)[n:])
        except ValueError:
            pass
    mol.rebuild()


# ------------------------------------------------------------- additions
def _solvent_chain(mol, prefer_water=True):
    for ch in mol.st[0]:
        if len(ch) and all(r.name in WATER for r in ch) == prefer_water and ch[0].het_flag == "H":
            return ch
    return None


def _new_chain_name(mol, prefer="W"):
    used = {ch.name for ch in mol.st[0]}
    for c in prefer + "XYZSLKJIHGFEDCBA0123456789":
        if c not in used:
            return c
    raise RuntimeError("no free chain id")


def _next_num(ch):
    return max((r.seqid.num for r in ch), default=0) + 1


def add_single_atom(mol, xyz, resname="HOH", element="O", atom_name=None, b=None, chain=None):
    """Water or ion at xyz. Goes into the water chain (or a new het chain)."""
    model = mol.st[0]
    ch = None
    if chain:
        ch = next((c for c in model if c.name == chain), None)
    if ch is None:
        ch = _solvent_chain(mol, prefer_water=True) if resname in WATER else None
    if ch is None:
        ch = model.add_chain(_new_chain_name(mol, "W" if resname in WATER else "S"))
        ch = model[ch.name] if not isinstance(ch, gemmi.Chain) else ch
    res = gemmi.Residue()
    res.name = resname
    res.seqid = gemmi.SeqId(_next_num(ch), " ")
    res.het_flag = "H"
    res.entity_type = gemmi.EntityType.Water if resname in WATER else gemmi.EntityType.NonPolymer
    at = gemmi.Atom()
    at.name = atom_name or (element.upper() if resname not in WATER else "O")
    at.element = gemmi.Element(element)
    at.pos = gemmi.Position(*xyz)
    at.occ = 1.0
    at.b_iso = b if b is not None else _local_b(mol, xyz)
    res.add_atom(at)
    ch.add_residue(res)
    mol.rebuild()
    return {"chain": ch.name, "resseq": res.seqid.num}


def _local_b(mol, xyz, r=6.0):
    if len(mol.xyz) == 0:
        return 30.0
    tree = cKDTree(mol.xyz)
    idx = tree.query_ball_point(xyz, r)
    return float(np.mean([mol.info[i][7] for i in idx])) if idx else 30.0


def find_water_sites(mol, xmap, sigma=3.0, dmin=2.4, dmax=3.3, max_n=500):
    """Difference-map peaks at H-bond distance from N/O and not clashing with any atom."""
    from .maps import SymMapper

    if not len(mol.xyz):
        return []
    polar = np.array([e in ("N", "O") for e in mol.elements])
    heavy = np.array([e != "H" for e in mol.elements])
    any_map = SymMapper(mol.st, mol.xyz, mask=heavy)
    polar &= heavy
    tree_all = any_map.tree
    out = []
    for xyz, h in xmap.peaks(sigma):
        pos, dist_any, _ = any_map.nearest(xyz)
        if dist_any < dmin - 0.2 or dist_any > dmax:
            continue
        d_pol = float(np.min(np.linalg.norm(mol.xyz[polar] - pos, axis=1))) if polar.any() else 99
        if not dmin <= d_pol <= dmax:
            continue
        if any(np.linalg.norm(np.array(o["xyz"]) - pos) < 1.0 for o in out):
            continue
        out.append({"xyz": [round(float(v), 3) for v in pos], "height_sigma": round(h, 2),
                    "dist_polar": round(d_pol, 2)})
        if len(out) >= max_n:
            break
    return out


def add_ligand(mol, cif_path, xyz, comp_id=None):
    res, comp = ideal_residue(comp_id or "", cif_path) if comp_id else ideal_residue_any(cif_path)
    pts = np.array([a.pos.tolist() for a in res if a.element.name != "H"])
    shift = np.asarray(xyz) - pts.mean(0)
    model = mol.st[0]
    ch = next((c for c in model if len(c) and c[0].het_flag == "H" and not all(r.name in WATER for r in c)), None)
    if ch is None:
        ch = model.add_chain(_new_chain_name(mol, "L"))
        ch = model[ch.name] if not isinstance(ch, gemmi.Chain) else ch
    new = gemmi.Residue()
    new.name = comp
    new.seqid = gemmi.SeqId(_next_num(ch), " ")
    new.het_flag = "H"
    new.entity_type = gemmi.EntityType.NonPolymer
    b = _local_b(mol, xyz)
    for a in res:
        if a.element.name == "H":
            continue
        c = a.clone()
        c.pos = gemmi.Position(*(np.array(a.pos.tolist()) + shift))
        c.occ, c.b_iso = 1.0, b
        new.add_atom(c)
    ch.add_residue(new)
    mol.rebuild()
    return {"chain": ch.name, "resseq": new.seqid.num, "comp_id": comp}


def ideal_residue_any(cif_path):
    doc = gemmi.cif.read(cif_path)
    names = [b.name[5:] for b in doc if b.name.startswith("comp_") and b.name != "comp_list"]
    if not names:
        raise ValueError(f"{cif_path}: no data_comp_XXX block")
    return ideal_residue(names[0], cif_path)


def _place(a, b, c, bond, angle, torsion):
    """NeRF: position of d given a, b, c and |cd|, angle(b,c,d), torsion(a,b,c,d)."""
    bc = c - b
    bc /= np.linalg.norm(bc)
    n = np.cross(b - a, bc)
    n /= np.linalg.norm(n)
    m = np.cross(n, bc)
    ang, tor = np.radians(angle), np.radians(torsion)
    d2 = np.array([-bond * np.cos(ang), bond * np.sin(ang) * np.cos(tor), bond * np.sin(ang) * np.sin(tor)])
    return c + d2[0] * bc + d2[1] * m + d2[2] * n


def add_residue(mol, chain, terminus="C", resname="ALA", phi=-60.0, psi=-45.0):
    """Extend a protein chain at the N or C terminus with ideal geometry (default helical phi/psi)."""
    model = mol.st[0]
    ch = next(c for c in model if c.name == chain)
    poly = [r for r in ch if r.het_flag != "H" and r.find_atom("CA", "*")]
    if not poly:
        raise ValueError("chain has no protein residues")
    P = lambda r, n: np.array(r.find_atom(n, "*").pos.tolist())
    if terminus.upper().startswith("C"):
        prev = poly[-1]
        n0, ca0, c0 = P(prev, "N"), P(prev, "CA"), P(prev, "C")
        N = _place(n0, ca0, c0, 1.329, 116.2, psi)
        CA = _place(ca0, c0, N, 1.458, 121.7, 180.0)
        C = _place(c0, N, CA, 1.525, 111.2, phi)
        O_prev = _place(N, ca0, c0, 1.231, 120.5, 180.0)
        for i in range(len(prev) - 1, -1, -1):
            if prev[i].name == "OXT":
                del prev[i]
        o = prev.find_atom("O", "*")
        if o:
            o.pos = gemmi.Position(*O_prev)
        N_next = _place(N, CA, C, 1.329, 116.2, psi)
        O = _place(N_next, CA, C, 1.231, 120.5, 180.0)
        num = prev.seqid.num + 1
    else:
        nxt = poly[0]
        n1, ca1, c1 = P(nxt, "N"), P(nxt, "CA"), P(nxt, "C")
        C = _place(c1, ca1, n1, 1.329, 121.7, phi)
        CA = _place(ca1, n1, C, 1.525, 116.2, 180.0)
        O = _place(ca1, n1, C, 1.231, 123.0, 0.0)
        N = _place(n1, C, CA, 1.458, 111.2, psi)
        num = nxt.seqid.num - 1
    template = poly[-1] if terminus.upper().startswith("C") else poly[0]
    b = float(np.mean([a.b_iso for a in template]))
    res = gemmi.Residue()
    res.name = "GLY"
    res.seqid = gemmi.SeqId(num, " ")
    res.entity_type = gemmi.EntityType.Polymer
    for name, pos, el in (("N", N, "N"), ("CA", CA, "C"), ("C", C, "C"), ("O", O, "O")):
        a = gemmi.Atom()
        a.name, a.element, a.pos, a.occ, a.b_iso = name, gemmi.Element(el), gemmi.Position(*pos), 1.0, b
        res.add_atom(a)
    if terminus.upper().startswith("C"):
        idx = [i for i, r in enumerate(ch) if r is poly[-1] or (r.seqid.num == poly[-1].seqid.num and r.name == poly[-1].name)][0]
        ch.add_residue(res, idx + 1)
    else:
        idx = [i for i, r in enumerate(ch) if r.seqid.num == poly[0].seqid.num and r.name == poly[0].name][0]
        ch.add_residue(res, idx)
    mol.rebuild()
    if resname.upper() != "GLY":
        mutate(mol, chain, num, resname)
    return {"chain": chain, "resseq": num}


# ------------------------------------------------------------ bookkeeping
def renumber(mol, chain, start, end, offset):
    ch = next(c for c in mol.st[0] if c.name == chain)
    moving = [r for r in ch if start <= r.seqid.num <= end]
    staying = {(r.seqid.num, r.seqid.icode) for r in ch if not (start <= r.seqid.num <= end)}
    for r in moving:
        if (r.seqid.num + offset, r.seqid.icode) in staying:
            raise ValueError(f"renumbering would clash with existing residue {chain}/{r.seqid.num + offset}")
    for r in moving:
        r.seqid = gemmi.SeqId(r.seqid.num + offset, r.seqid.icode)
    mol.rebuild()
    return len(moving)


def delete(mol, chain, start, end=None, atom=None, altloc=None, side_chain_only=False):
    end = start if end is None else end
    ch = next(c for c in mol.st[0] if c.name == chain)
    n = 0
    for ri in range(len(ch) - 1, -1, -1):
        res = ch[ri]
        if not start <= res.seqid.num <= end:
            continue
        if atom is None and altloc is None and not side_chain_only:
            del ch[ri]
            n += 1
            continue
        for ai in range(len(res) - 1, -1, -1):
            a = res[ai]
            if atom and a.name != atom:
                continue
            if altloc and a.altloc != altloc:
                continue
            if side_chain_only and a.name in MAIN_CHAIN | {"CB"}:
                continue
            del res[ai]
            n += 1
    mol.rebuild()
    return n


def set_occupancy(mol, chain, start, end, occ, altloc=None):
    for ch, ri, res in mol.residues_in_range(chain, start, end):
        for a in res:
            if altloc is None or a.altloc == altloc:
                a.occ = occ
    mol.rebuild()


def set_bfactor(mol, chain, start, end, b):
    for ch, ri, res in mol.residues_in_range(chain, start, end):
        for a in res:
            a.b_iso = b
    mol.rebuild()


def flip_peptide(mol, chain, num):
    """Rotate the C=O of residue num / N-H of num+1 by 180 deg about the CA-CA axis."""
    ch, ri, res = mol.find_residue(chain, num)
    _, _, nxt = mol.find_residue(chain, num + 1)
    ca1 = np.array(res.find_atom("CA", "*").pos.tolist())
    ca2 = np.array(nxt.find_atom("CA", "*").pos.tolist())
    for r, names in ((res, ("C", "O")), (nxt, ("N", "H"))):
        for a in r:
            if a.name in names:
                a.pos = gemmi.Position(*_rotate(np.array([a.pos.tolist()]), ca1, ca2 - ca1, 180)[0])
    mol.rebuild()


# ------------------------------------------------------------- ligands
# Crystallisation / cryo / buffer components that are almost never the ligand of interest.
ADDITIVES = set("""
EDO GOL PEG PGE PG4 1PE P6G PE4 PE8 2PE 12P 15P XPE P33 PG0 PG5 7PE PE3 PE5 PG6 M2M
MPD MRD DMS DMF ACT ACY FMT EOH IPA PGO PGR BU3 BU1 PDO 1BO TBU MOH ETX
TRS EPE MES BTB CIT FLC TAR TLA MLI MLA SIN CAC B3P MOPS MPO PIP HEPES NHE CXS CHES
SO4 PO4 PI 2HP NO3 SCN AZI IOD BR CL NA K CS LI RB NH4 F
UNX UNL BME DTT DTU TCE IMD HEZ BCN SGM CO2 OXY PER
BOG LMT LDA C8E HTG DMU UMQ BNG LMU SDS CPS
OLC PLM DAO HEX HP6 OCT D10 D12 TRD C14 MYS R16 ACE
""".split())  # last line: lipids / alkanes / acetyl fragments modelled in the rhodopsin (7ZBC-based) structures

METALS = {"MG", "CA", "ZN", "MN", "FE", "FE2", "CO", "NI", "CU", "CU1", "CD", "HG", "MO", "W", "SR", "BA", "YB", "GD"}


def list_ligands(mol):
    """Non-polymer, non-water residues with a guess whether each is biologically relevant.

    Returns [{chain, resseq, icode, resname, n_atoms, relevant, reason, center}]."""
    from scipy.spatial import cKDTree

    polar = np.array([e in ("N", "O", "S") for e in mol.elements])
    water = np.array([i[3] in WATER for i in mol.info])
    tree = cKDTree(mol.xyz) if len(mol.xyz) else None
    out = []
    for ch in mol.st[0]:
        for res in ch:
            if res.name in WATER:
                continue
            # polymer residues are skipped, except non-standard ones (e.g. LYR = retinal-Lys296), which are always relevant
            nonstandard = False
            if res.entity_type == gemmi.EntityType.Polymer:
                info = gemmi.find_tabulated_residue(res.name)
                if info is not None and info.is_standard():
                    continue
                nonstandard = True
            heavy = [a for a in res if a.element.name != "H"]
            if not heavy:
                continue
            pts = np.array([a.pos.tolist() for a in heavy])
            name = res.name.upper()
            if nonstandard:
                relevant, reason = True, "non-standard residue"
            elif name in ADDITIVES:
                relevant, reason = False, "common additive/buffer"
            elif name in METALS or (len(heavy) == 1 and gemmi.Element(heavy[0].element.name).is_metal):
                # metal: relevant when coordinated by >= 2 non-water N/O/S
                key = (ch.name, res.seqid.num, res.seqid.icode.strip())
                n = 0
                for j in tree.query_ball_point(pts[0], 2.8):
                    if polar[j] and not water[j] and mol.keys[j][:3] != key:
                        n += 1
                relevant, reason = (n >= 2), f"metal, {n} non-water ligand atoms"
            else:
                relevant, reason = True, "ligand"
            out.append({"chain": ch.name, "resseq": res.seqid.num, "icode": res.seqid.icode.strip(),
                        "resname": res.name, "n_atoms": len(heavy), "relevant": relevant, "reason": reason,
                        "center": [round(float(v), 3) for v in pts.mean(0)]})
    return out
