"""Molecule container: a gemmi Structure plus flat atom arrays for the viewer."""
from __future__ import annotations

import itertools
import os

import gemmi
import numpy as np
from scipy.spatial import cKDTree

BACKBONE = {"N", "CA", "C", "O", "OXT", "H", "HA"}
WATER = {"HOH", "WAT", "DOD"}


# bond orders as sent to the viewer
SINGLE, DOUBLE, TRIPLE, AROMATIC, DELOC = 1, 2, 3, 4, 5
_ORDER = {"SING": SINGLE, "DOUB": DOUBLE, "TRIP": TRIPLE, "AROM": AROMATIC, "DELO": DELOC}


class BondOrders:
    """Bond orders per residue type, from restraint dictionaries.

    Looks in the session's ligand dictionaries first (``paths`` is the live list
    the session appends to), then in TorchRef's monomer library."""

    def __init__(self, paths: list[str] | None = None):
        self.paths = paths if paths is not None else []
        self._cache: dict[str, dict] = {}
        self._seen_paths: set[str] = set()

    def _library_file(self, resname):
        import importlib.util
        spec = importlib.util.find_spec("torchref")  # locate without importing (torch import is slow)
        if spec is None or not spec.submodule_search_locations:
            return None
        p = os.path.join(list(spec.submodule_search_locations)[0], "data", "monomer_library",
                         resname[0].lower(), resname.upper() + ".cif")
        return p if os.path.exists(p) else None

    @staticmethod
    def _parse(block):
        out = {}
        for tag in ("type", "value_order"):
            t = block.find("_chem_comp_bond.", ["atom_id_1", "atom_id_2", tag])
            if len(t):
                for r in t:
                    o = _ORDER.get(gemmi.cif.as_string(r[2]).upper()[:4], SINGLE)
                    a, b = gemmi.cif.as_string(r[0]), gemmi.cif.as_string(r[1])
                    out[(a, b)] = out[(b, a)] = o
                break
        return out

    def _scan_new_paths(self):
        for p in self.paths:
            if p in self._seen_paths:
                continue
            self._seen_paths.add(p)
            try:
                for b in gemmi.cif.read(p):
                    if b.name.startswith("comp_") and b.name != "comp_list":
                        self._cache[b.name[5:]] = self._parse(b)
            except Exception:
                pass

    def table(self, resname):
        self._scan_new_paths()
        if resname not in self._cache:
            tab = {}
            f = self._library_file(resname)
            if f:
                try:
                    b = gemmi.cif.read(f).find_block("comp_" + resname)
                    if b is not None:
                        tab = self._parse(b)
                except Exception:
                    pass
            self._cache[resname] = tab
        return self._cache[resname]


def cov_radius(el: str) -> float:
    return gemmi.Element(el).covalent_r if el else 0.77


class Molecule:
    _ids = itertools.count(1)

    def __init__(self, path: str | None = None, structure: gemmi.Structure | None = None, name=None,
                 bond_orders: BondOrders | None = None):
        self.id = next(self._ids)
        self.path = path
        if structure is None:
            structure = gemmi.read_structure(path)
        structure.setup_entities()
        self.st = structure
        self.name = name or (os.path.basename(path) if path else f"mol{self.id}")
        self.visible = True
        self.undo_stack: list[gemmi.Structure] = []
        self.redo_stack: list[gemmi.Structure] = []
        self.fixed: set[tuple] = set()  # atom keys fixed during refinement
        self.bond_orders = bond_orders or BondOrders()
        self.rebuild()

    # ------------------------------------------------------------------ keys
    @staticmethod
    def res_key(ch: gemmi.Chain, res: gemmi.Residue):
        return (ch.name, res.seqid.num, res.seqid.icode.strip())

    @staticmethod
    def atom_key(ch, res, at):
        return (ch.name, res.seqid.num, res.seqid.icode.strip(), at.name, at.altloc if at.altloc != "\0" else "")

    # ------------------------------------------------------------ flat arrays
    def rebuild(self):
        """Recompute flat atom table + bonds after any edit."""
        self.st.remove_empty_chains()
        model = self.st[0]
        refs, keys, xyz, el, info = [], [], [], [], []
        for ch in model:
            for ri, res in enumerate(ch):
                for at in res:
                    refs.append((ch.name, ri, at))
                    keys.append(self.atom_key(ch, res, at))
                    xyz.append(at.pos.tolist())
                    el.append(at.element.name)
                    info.append((ch.name, res.seqid.num, res.seqid.icode.strip(), res.name, at.name,
                                 keys[-1][4], at.occ, at.b_iso, res.het_flag == "H"))
        self.keys = keys
        self.key_index = {k: i for i, k in enumerate(keys)}
        self.xyz = np.array(xyz, dtype=np.float64).reshape(-1, 3)
        self.elements = el
        self.info = info
        self.bonds = self._bonds()
        self.orders = self._orders()

    def _bonds(self):
        n = len(self.xyz)
        if n < 2:
            return np.zeros((0, 2), np.int32)
        tree = cKDTree(self.xyz)
        pairs = tree.query_pairs(2.3, output_type="ndarray")
        if len(pairs) == 0:
            return np.zeros((0, 2), np.int32)
        rad = np.array([cov_radius(e) for e in self.elements])
        d = np.linalg.norm(self.xyz[pairs[:, 0]] - self.xyz[pairs[:, 1]], axis=1)
        ok = d < (rad[pairs[:, 0]] + rad[pairs[:, 1]] + 0.45)
        alt = np.array([k[4] for k in self.keys])
        a0, a1 = alt[pairs[:, 0]], alt[pairs[:, 1]]
        ok &= (a0 == "") | (a1 == "") | (a0 == a1)
        resn = np.array([i[3] for i in self.info])
        ok &= ~np.isin(resn[pairs[:, 0]], list(WATER)) & ~np.isin(resn[pairs[:, 1]], list(WATER))
        h = np.array([e == "H" for e in self.elements])
        ok &= ~(h[pairs[:, 0]] & h[pairs[:, 1]])
        return pairs[ok].astype(np.int32)

    def _orders(self):
        """Bond order for each bond: from the residue's dictionary when both atoms are in one residue."""
        out = np.ones(len(self.bonds), np.int8)
        info = self.info
        for k, (i, j) in enumerate(self.bonds):
            a, b = info[i], info[j]
            if a[:3] != b[:3] or a[3] != b[3]:
                continue
            o = self.bond_orders.table(a[3]).get((a[4], b[4]))
            if o:
                out[k] = o
        return out

    def to_json(self):
        return {
            "id": self.id, "name": self.name, "path": self.path, "visible": self.visible,
            "xyz": np.round(self.xyz, 3).ravel().tolist(),
            "element": self.elements,
            "atoms": [list(i[:6]) + [round(i[6], 2), round(i[7], 1), i[8]] for i in self.info],
            "bonds": self.bonds.ravel().tolist(),
            "orders": self.orders.tolist(),
            "fixed": [self.key_index[k] for k in self.fixed if k in self.key_index],
            "cell": list(self.st.cell.parameters) if self.st.cell.is_crystal() else None,
            "undo": len(self.undo_stack), "redo": len(self.redo_stack),
        }

    # -------------------------------------------------------------- helpers
    def find_residue(self, chain, num, icode=""):
        for ch in self.st[0]:
            if ch.name != chain:
                continue
            for ri, res in enumerate(ch):
                if res.seqid.num == num and res.seqid.icode.strip() == (icode or ""):
                    return ch, ri, res
        raise KeyError(f"residue {chain}/{num}{icode} not found")

    def residues_in_range(self, chain, start, end):
        out = []
        for ch in self.st[0]:
            if ch.name == chain:
                for ri, res in enumerate(ch):
                    if start <= res.seqid.num <= end:
                        out.append((ch, ri, res))
        return out

    def residue_center(self, chain, num, icode=""):
        _, _, res = self.find_residue(chain, num, icode)
        ca = res.find_atom("CA", "*") or res[0]
        return ca.pos.tolist()

    def residue_extent(self, chain, num, icode=""):
        """(centroid of the heavy atoms of the residue, largest distance of any of its atoms from that centroid)."""
        _, _, res = self.find_residue(chain, num, icode)
        pts = np.array([a.pos.tolist() for a in res if a.element.name != "H"] or [res[0].pos.tolist()])
        c = pts.mean(0)
        return c.tolist(), float(np.linalg.norm(pts - c, axis=1).max())

    def neighbours_of_residues(self, residue_keys, radius):
        """Residue keys with any atom within radius of any atom of the given residues."""
        idx = [i for i, k in enumerate(self.keys) if k[:3] in residue_keys]
        if not idx:
            return set()
        tree = cKDTree(self.xyz)
        hits = tree.query_ball_point(self.xyz[idx], radius)
        out = set()
        for h in hits:
            for j in h:
                out.add(self.keys[j][:3])
        return out

    def checkpoint(self):
        self.undo_stack.append(self.st.clone())
        del self.undo_stack[:-40]
        self.redo_stack.clear()

    def undo(self):
        if not self.undo_stack:
            return False
        self.redo_stack.append(self.st.clone())
        self.st = self.undo_stack.pop()
        self.rebuild()
        return True

    def redo(self):
        if not self.redo_stack:
            return False
        self.undo_stack.append(self.st.clone())
        self.st = self.redo_stack.pop()
        self.rebuild()
        return True

    def write(self, path):
        self.st.setup_entities()
        if path.endswith((".cif", ".mmcif")):
            doc = self.st.make_mmcif_document()
            doc.write_file(path)
        else:
            self.st.write_pdb(path)
        return path
