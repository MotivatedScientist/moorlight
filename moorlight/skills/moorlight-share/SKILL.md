---
name: moorlight-share
description: Make a self-contained, phone-friendly viewer page for a structure and its maps (models + density/difference maps + preset slice planes) that needs no server, for publishing as a claude.ai artifact or sending as a file. One command from files, a refinement folder or a PDB id. Works for small molecules, proteins and EM maps; inputs PDB/mmCIF and CCP4/MRC maps or MTZ map coefficients. Use when the user wants to look at a structure or map from a phone, a Remote Control session or a link, or asks for planes or slices through a structure.
---

# Shareable structure viewer pages

One command turns a structure and its maps into one self-contained HTML page and prints what the page shows. The
whole job should take about 30 s: run the command, publish, reply. You never write or edit JSON.

## Fast path

1. **Run one command** (about 3–5 s):

   ```
   moorlight-share INPUTS --title "Two to four words" [--center SPECS] [--plane "NAME=SPECS[; option=value]" ...] -o DIR/page.html
   ```

   - **INPUTS** is one of:
     - a refinement folder. Moorlight recognises:
       - TorchRef difference refinement: dark and light models, the q-weighted light − dark map, the extrapolated light map, and the dark 2mFo-DFc and mFo-DFc maps;
       - TorchRef refinement output;
       - any folder with one model and its maps.
     - model and map files (`.pdb/.cif`, `.mtz/.ccp4/.map/.mrc`), optionally named: `dark.pdb=dark`, `10ps.ccp4="10 ps"`.
     - a PDB id (`3hs4`), which downloads the PDB-REDO model and maps into DIR.
   - **Atom specs** come in these forms:
     - `A/85 CA` is an atom; `A/85` is a residue (all its heavy atoms);
     - `2#A/85 SG` is an atom in model 2 (the default is model 1);
     - `A/85 CA:B` is altloc B;
     - a residue name (`RET`, `ZN`, `AZM C1`) also works. If there are several copies, the most buried one is used.

     Use them directly; you don't need to look anything up first.
   - **What it prints:**
     - the page size and the starting view;
     - each map (shown or hidden, level, rms);
     - the strongest difference features near the centre, with their nearest atoms (`A/93 LEU CG`);
     - every plane with its fit, and for each difference map: the signal at the plane's own atoms (rms, min, max),
       which tells which group changes most, and the features in the slice.

     That is the content of your reply.
2. **Publish** DIR/page.html with the Artifact tool: icon "molecule", a one-sentence description. For a file to
   e-mail or attach, build with `--standalone` instead (or as well).
3. **Reply** with what the summary says, plus the link.
   - Don't re-check the numbers with other tools, read the moorlight source or write scripts.
   - The printed summary and `DIR/page.check.txt` hold every number the page shows.

**Follow-ups:** for "add a slice…", "open on…", "show the other map", "label…", run
`moorlight-share again DIR/page.spec.json [more options]` and republish the same file path, which keeps the link.
- Single options given later win.
- `--plane` with a new name adds a plane; with an existing name it replaces that plane in place (e.g. to add
  `copies`). `--no-plane NAME` removes one.
- `--label` and `--distance` add to what is already there; `--no-label SPECS` and `--no-distance "A, B"` remove.

## Options

| Option | Effect |
|---|---|
| `--title`, `--description` | page name (two to four words, no "Name: explanation") and one line under it |
| `--center SPECS` | what the page opens on, face-on (comma-separated specs). A metal gets its ligands labelled, with distances. Default: the `--plane` atoms, else the strongest difference feature, else the most buried ligand |
| `--radius R` | Å from the centre to the screen edge at the start |
| `--plane "NAME=SPECS; ..."` | a preset slice plane (repeatable). Three or more atoms (or a residue) give their best plane; one metal atom (`Zn site=ZN`) gives its coordination plane; one other atom gives the plane through it and its bonded neighbours |
| `--no-plane NAME` | remove a plane (with `again`) |
| `--label SPECS`, `--distance "A, B"` | labels and distance lines at the start (repeatable) |
| `--show MAP`, `--hide MAP` | which maps are on at the start. MAP is the name or a unique part of it. By default one density and one difference map are on |
| `--drop MAP` | leave a map out (smaller page) |
| `--level 4`, `--level "MAP=1.2"` | contour level: of all difference maps, or of one map |
| `--units absolute --unit-label SD` | maps already in their own units, e.g. noise SD. Give `--level` too (e.g. 4) |
| `--standalone` | a full HTML document to send as a file. It opens offline in any current browser |
| `--around R` | store the maps only within about R Å of the starting centre. Much smaller files (e.g. `--around 15` for e-mail). Density is then missing further out, and automatic planes stay inside it. For large structures it also keeps the maps on their own grid: without it they may be resampled more coarsely, which the summary notes, and peak values can be off by about 0.5 σ |
| `--at "SPECS"`, `--near SPEC [--near-radius R]` | also print map values at atoms, or the atoms (one line per residue) and peaks around a point |
| `--planes N`, `--min L` | automatic planes when no `--plane` is given (default 5, at features beyond 1.5 × the level) |

**Plane options**, after `;` in `--plane`:

| Option | Effect |
|---|---|
| `copies=-1,1` | parallel copies at these offsets (Å), named "NAME −1 Å" and "NAME +1 Å" |
| `toward=SPEC` | the side the normal points to, and so where + offsets and copies go. Give an atom well off the plane, e.g. one the change moves towards |
| `perpendicular-to=PLANE` | a plane perpendicular to PLANE (an earlier plane's name, or atoms). It contains the line through NAME's atoms and passes through `center=`. For a curved chain, give the two end atoms of the stretch you mean (`RET C7, A/216 NZ`), not every atom |
| `thickness=2` | a slab (Å, full width) that shows the strongest value within it |
| `center=SPEC` | moves the plane to pass through this atom |
| `offset=1` | moves the plane along its normal (Å) |
| `axis=A, B` | its horizontal direction on screen |
| `map=NAME`, `level=L` | the map and level the plane starts with |

A coordination plane with five or more ligands goes through the four that lie flattest with the metal (its
equatorial plane), preferring ligands of the metal's own residue: a haem iron gives the porphyrin plane. A non-planar site, such as a tetrahedral zinc, gets a slab thick enough to take in all its
ligands.

**Examples:**

```
moorlight-share /path/to/torchref_difference_run --title "Light minus dark" --center "RET, A/216 NZ" \
  --plane "Retinal=RET C5, RET C7, RET C9, RET C11, RET C13, RET C15, A/216 NZ; copies=-1,1" \
  --plane "Across the chain=RET C7, A/216 NZ; perpendicular-to=Retinal; center=A/216 NZ; toward=RET C20" \
  -o out/br.html
moorlight-share 3hs4 --title "Carbonic anhydrase acetazolamide" --center ZN --plane "Zn site=ZN" \
  --plane "Ring=AZM C1, AZM N3, AZM N2, AZM C2, AZM S2" -o out/ca2.html --standalone
moorlight-share dark.pdb=dark excited.pdb=excited 10ps.ccp4="10 ps" 1ns.ccp4="1 ns" --units absolute --unit-label SD \
  --level 4 --plane "Cu site=A/1 CU" --plane "Ring=A/401 C1, A/401 C2, A/401 C3, A/401 C4; copies=-2,-1,1,2" -o out/compare.html
moorlight-share again out/br.spec.json --plane "Asp85=A/85" --label "A/85 OD1, A/85 OD2"
```

**More detail when you need it:**
- `moorlight-share list MODEL [RESIDUE ...]` prints:
  - chains and residue ranges;
  - ligands;
  - metals with their ligand atoms and distances;
  - waters;
  - alternate conformers.

  With residues (`RET`, `A/94`), it prints their atom names instead.
- `DIR/page.check.txt` holds every number the page shows: all peaks with ranks, each plane's atoms with heights, and values at labels.
- **No browser:** a machine without a browser cannot render the page. Say so if you could not look at it.

## Maps

- **MTZ files:** an MTZ gives the maps a refinement program writes for viewing (2mFo-DFc and mFo-DFc: FWT/DELFWT,
  2FOFCWT/FOFCWT, ...).
  - `file.mtz=name:F,PHI` picks columns.
  - `F*W,PHI` weights the amplitudes by a weight column.
- **TorchRef `difference_data.mtz`:**
  - It gives light − dark dF on the dark phases, q-weighted by default (`dF*W_Q,PHDELWT`).
  - Its FWT/PHWT is the extrapolated light map, 2Fext − Fc, not a 2mFo-DFc.
  - Levels are in σ, the rms of the whole map, signal included, so a TorchRef light − dark map has no separate noise scale here.
- **Map kind:** `diff` maps are green (+) and red (−); `2fofc` maps are blue. The kind comes from the MTZ columns or the file name, or failing that from the values: a density map has almost nothing below −2.5 rms.
- **Default levels:** 1.5 σ for density maps, ±3 σ for difference maps.
- **Grid:** each map is stored on its own grid, so values and peak heights match Coot.
  - Small cells are stored whole and periodic.
  - Large cells and EM boxes are stored as a block around the model with a 5 Å margin.
  - Only if neither fits is the map resampled.
  - Pages are about 0.4 MB per map for small cells and 2–5 MB for two protein maps. Artifact pages must stay at or below 16 MB, and e-mail is safest below about 7 MB.
- **Values at atoms** are trilinear interpolation on the stored grid: the same number a tap on the page shows.
- **Peaks** are grid points that are the extreme of their 3×3×3 neighbourhood.
- **Neighbour copies:** neighbour-cell and symmetry copies within 5 Å of the model are drawn in grey and named with their operator, e.g. `A/219 O [-x,y+1/2,-z+1/2 -b]` or `S/14 O [-a]`. Specs accept the same text.

## The page

- **Maps:** each map has an on/off switch and a log-scale level slider, with its rms and the level on its own scale.
  - Two-finger scroll, or the mouse wheel, sets the level of the map in use: the first one switched on.
- **Mesh or Slice:** maps are drawn as Mesh or as Slice, and changing maps never changes the mode.
  - Slice mode draws only slices: the live plane through the centre, and the preset and saved planes.
  - Every slice shows the map in use at its slider level. With every map off, the planes are drawn empty.
  - Tapping a plane's name looks at it face-on.
  - The viewer can tilt, thicken, fix, save and copy planes.
- **Structure:**
  - each model and the neighbour cells can be switched on or off;
  - a hydrogens switch;
  - Hide in front (on by default) hides residues in front of the centre residue, together with their density;
  - clear labels and Reset view.
- **Controls:**

  | Input | Action |
  |---|---|
  | drag | rotate |
  | two-finger click-drag (right-drag) | pan |
  | pinch / Ctrl+scroll | zoom |
  | tap an atom | label and map values |
  | double-tap an atom | centre on it |

  The depth bar on the right moves the visible layer. Full screen is F; landscape puts the panel beside the view.
- **Self-contained:** the page fetches nothing. three.js 0.170.0 (MIT) is written into it, and it uses system fonts.
  - Desktop and Android browsers open the file.
  - An iPhone's file preview may not run it, so send iPhone users the link.
  - "About this file" lists the input files with SHA-256 checksums, the Moorlight version and commit, and the build time.

## Spec files (rarely needed)

`DIR/page.spec.json` records the page: the files, maps, view, planes and the command line that made it. The other commands that work on it:
- `moorlight-share again` re-runs that command line.
- `moorlight-share SPEC.json [-o page.html] [--check [--at ...] [--near ...]]` builds or checks a spec directly.
- `moorlight-share init FILES -o spec.json` drafts one.

Prefer options and `again` to editing a spec.
