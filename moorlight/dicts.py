"""Restraint dictionaries embedded in refinement output (phenix mmCIF data_comp_XXX blocks).

They are rewritten in the minimal CCP4 layout TorchRef parses reliably: primed atom names
double-quoted, only atom/bond/angle/tor/plane/chir loops kept (descriptor loops break the reader).
"""
from __future__ import annotations

import os

import gemmi

KEEP = ["_chem_comp_atom.", "_chem_comp_bond.", "_chem_comp_angle.", "_chem_comp_tor.",
        "_chem_comp_plane_atom.", "_chem_comp_chir."]


def _q(v):
    v = gemmi.cif.as_string(v)
    return f'"{v}"' if ("'" in v or " " in v or v == "") else v


def comp_ids(path):
    try:
        doc = gemmi.cif.read(path)
    except Exception:
        return []
    return [b.name[5:] for b in doc if b.name.startswith("comp_") and b.name != "comp_list"]


def extract_all(path, outdir):
    """Write every data_comp_XXX block of a CIF as its own TorchRef-safe dictionary. Returns paths."""
    doc = gemmi.cif.read(path)
    groups = {}
    for b in doc:
        for row in b.find("_chem_comp.", ["id", "group"]):
            groups[gemmi.cif.as_string(row[0])] = gemmi.cif.as_string(row[1])
    os.makedirs(outdir, exist_ok=True)
    out = []
    for b in doc:
        if not b.name.startswith("comp_") or b.name == "comp_list":
            continue
        comp = b.name[5:]
        atoms = b.find("_chem_comp_atom.", ["atom_id", "type_symbol"])
        if not len(atoms) or not b.find_mmcif_category("_chem_comp_bond."):
            continue  # single atoms (ions) need no dictionary, and TorchRef rejects bond-less files
        nh = sum(1 for r in atoms if gemmi.cif.as_string(r[1]).upper() != "H")
        group = groups.get(comp, "non-polymer") or "non-polymer"
        lines = ["data_comp_list", "loop_", "_chem_comp.id", "_chem_comp.three_letter_code", "_chem_comp.name",
                 "_chem_comp.group", "_chem_comp.number_atoms_all", "_chem_comp.number_atoms_nh",
                 "_chem_comp.desc_level", f"{comp} {comp} {comp} {_q(group)} {len(atoms)} {nh} .", "",
                 f"data_comp_{comp}"]
        for pre in KEEP:
            tab = b.find_mmcif_category(pre)
            if not tab:
                continue
            lines.append("loop_")
            lines += [t for t in tab.tags]
            for r in tab:
                lines.append(" ".join(_q(r[i]) for i in range(len(r))))
        p = os.path.join(outdir, f"{comp}.cif")
        with open(p, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        out.append(p)
    return out
