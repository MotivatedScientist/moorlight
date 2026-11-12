# Moorlight

A small Coot 0.9-style model-building viewer that runs in the browser, with real-space refinement
built on TorchRef restraints and Claude built in. Made for TorchRef (difference) refinement results.

```bash
pip install moorlight                 # pulls in TorchRef, gemmi, torch (CPU is fine)
moorlight path/to/torchref/run/     # refined.pdb + refined.mtz, or D.pdb + L_refined.pdb + difference_data.mtz
moorlight model.pdb maps.mtz lig.cif  # or individual files
moorlight 9DG2.pdb 9DG2.mtz:9DG2 9DCP.mtz:9DCP   # file.mtz:label names the maps
```

## What it does

- **Maps.** It recognises TorchRef columns automatically:
  - `FWT/PHWT` and `DELFWT/PHDELWT` from single-dataset refinement
  - `WDF`, `DF` and `DFc` with `PHIC_dark`, `mDFop-DFc/PHIC_diff`, and the extrapolated maps (with `PHIC_light`) from difference refinement
  - phenix `2FOFCWT`/`FOFCWT`, Fo-Fo (`FoFo/PHFc` from `phenix.fobs_minus_fobs_map`) and polder/omit maps
  - other `F/PHI` column pairs, or CCP4 maps

  `load_mtz` takes a `label` (map name), explicit `columns` (`F,PHI[,kind[,name]]`) and a `d_min` to compare data sets at
  matched resolution. Maps can be renamed (`rename_map`).

  Maps are drawn as Coot-style chicken wire within a 10 Å radius of the centre. The scroll wheel changes the level of the map marked ●.
- **Real-space refinement.** Choose a zone, the residue ±1 (R), a sphere (Shift-R), or a new alternative conformer. Each refinement builds TorchRef restraints (bonds, angles, torsions, planes, chirals; ligand CIFs included) for the moving residues plus a fixed shell around them. These are minimised against the refinement map (the one marked **R**) with live updates in the view. Drag an atom to pull it; Enter accepts, Esc rejects. Fixed atoms never move.
- **Difference target.** Set the refinement target to *difference* (or *map + difference*). Moving atoms are then fitted so that the model's light-minus-dark density matches a difference map (WDF by default), with a fitted scale. The dark reference is D.pdb for TorchRef difference refinement, otherwise the residue's other alternative conformers, otherwise the starting coordinates. The banner shows the live difference CC.
- **Repulsion.** Non-bonded repulsion is off by default, so you can pull atoms on top of each other. Tick *atoms repel* to switch it on.
- **Building.** Alternative conformers, rotamers ranked by density, mutate, pepflip, adding residues at either terminus, renumbering, deleting, waters (W, or *Find water sites*), ions, ligands from a CIF that contains coordinates, occupancy/B, undo/redo, save.
- **Peaks tab.** Lists the strongest ± difference peaks with their nearest atom, after symmetry mapping. Click a peak to go there.
  Each peak carries its rank among all peaks of that sign in the cell ("15th of 560 above 3.5σ").
- **Tours.** A tour is a list of stops, each a saved view (centre, rotation, zoom, slab, visible molecules and maps with their
  levels, labels and distances) plus a short text. Claude builds one with `tour_add` and `tour_save`. You can step through it
  with ◀ ▶ in the Display tab, and share it as a JSON file: `tour_load` opens the models and maps it needs. Single views can
  be saved with `save_view` / `restore_view`.
- **Claude tab.** A chat running headless Claude Code with every viewer command as a tool. Each message carries your current view (visible maps and levels, centre, selected atoms), so "this map" means what you're looking at. Claude can take screenshots of the view. The *shell* box lets it run commands such as TorchRef refinements; it's off by default.

Press `?` in the viewer for keys and mouse controls.

## Using it from any Claude Code session (MCP)

The viewer writes its address to `~/.moorlight/server.json`. To attach the tools to any Claude Code session:

```bash
claude mcp add moorlight -- python -m moorlight.mcp
```

Claude can then call `state`, `screenshot`, `goto`, `set_map`, `peaks`, `map_correlation`, `refine`, `add_altconf` and the other commands directly.

## Driving it from a shell or an already-running agent session

MCP servers cannot be added to a Claude Code session that is already running. The same commands are available from the shell:

```bash
moorlight cmd --list                                   # every command with its parameters
moorlight cmd state compact=true
moorlight cmd solo mol=2
moorlight cmd frame_atoms atoms='["A/381 SG", "B/901 C1"]'
moorlight cmd density_profile a="A/381 SG" b="B/901 C1" maps=3,4
moorlight cmd screenshot width=800
```

Values are read as JSON when possible (numbers, `true`, lists), otherwise as strings, and the result is printed as JSON. The viewer
is found through `--url`, `$MOORLIGHT_URL` or `~/.moorlight/server.json`, and you get an error if that file points at a viewer
that is no longer running. `moorlight --no-browser` prints `Moorlight ready: <url>` once the port is open.

Underneath is a small HTTP API:

| endpoint | |
|---|---|
| `GET /api/commands` | all commands, with descriptions and parameter types |
| `POST /api/cmd` | `{"cmd": "goto", "args": {"chain": "A", "resseq": 85}}` → `{"ok": true, "result": ...}` or `{"ok": false, "error": ...}` |
| `GET /api/state` | everything the browser needs to draw (models, maps, view, tour) |

**Atom specs.** Commands that take atoms accept `A/85 CA`, `A/85 LYS CA:B` (altloc B), `2#A/85 SG` (molecule 2), `A/85`
(residue centroid) or `{"chain": "A", "resseq": 85, "atom": "CA"}`.

**Commands for presenting and measuring:**
- **Display:** `set_molecule` (visible, colour, style, hydrogens), `solo`. The panel checkboxes use the same commands, so `state` reports what you see.
- **Framing:** `goto` (residue), `goto_atom`, `frame_atoms` (fits the atoms and lays the first pair horizontally in the screen plane), and `set_center` on an atom or the midpoint of several.
- **Measuring:**
  - `distance`, across molecules; without altlocs, every conformer pair.
  - `density_profile` along a line, with the lowest value relative to the ends.
  - `density_at_point`.
  - `density_at_atoms`, with `maps=[...]` independent of visibility.
- **Annotation:** `label_atoms`, `show_distance`, `clear_annotations`. Labels and distances are drawn into screenshots.
- **Screenshots:** `screenshot` waits until the view has settled and its map contours are computed. It takes `width`, `crop` and `path`.

## Claude skill: guided result tours

`moorlight --install-skill` copies the bundled `moorlight-result-tour` skill to `~/.claude/skills` (or give a directory). It tells Claude how to
walk you through results (difference maps, moving residues, waters): plan the stops from the analysis, show one map at a time, frame each target with
`goto` (residue plus ~5 Å context, depth slab), back every statement with `density_at_atoms` numbers and stop after each stop. The built-in chat uses
the same rules.

`goto` takes `radius` (Å of context around the residue, default 5; 0 = only centre) and `slab` (depth half-thickness in Å).

## Notes / limits (v0.1)

- Real-space refinement uses trilinear interpolation of the map and a simple non-bonded repulsion. Ramachandran restraints are not used yet.
- Hydrogens are left out of refinement and follow their parent atom afterwards.
- Ligands need coordinates in the CIF (from acedrg, grade or eLBOW). There's no RDKit conformer generation yet.
- Rotamers come from the Penultimate library (modal χ values), ranked by mean density at the side-chain atoms.
