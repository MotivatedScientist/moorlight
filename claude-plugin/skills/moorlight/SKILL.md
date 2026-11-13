---
name: moorlight
description: Install, open and drive Moorlight, a Coot 0.9-style crystallographic model viewer with TorchRef real-space refinement, inside the Claude Code session's built-in browser. Use when the user wants to look at a model with its maps (PDB/mmCIF + MTZ or CCP4 maps, TorchRef or phenix refinement output, difference maps), open "Moorlight" or "Coot", inspect density, compare data sets, or do real-space refinement and model building.
---

# Moorlight in the built-in browser

Moorlight is a Python server plus a browser viewer. **Always show it in this session's built-in browser pane**, never in the
user's system browser: start the server with `--no-browser`, then open its URL with the built-in browser tools.

## 1. Make sure it is installed

```bash
command -v moorlight || python -m pip show moorlight
```

If it is missing, install it into a Python 3.10+ environment (a dedicated conda env is best, because TorchRef pulls in torch):

```bash
conda create -y -n moorlight python=3.12 && conda run -n moorlight pip install moorlight
```

Then call it as `conda run -n moorlight moorlight …`, or put a small launcher on the PATH. Ask before creating environments. Check
the version with `moorlight cmd --help` (version 0.2 or later has `moorlight cmd`; upgrade with `pip install -U moorlight`).

## 2. Start the viewer and open it in the built-in browser

1. Start the server **in the background** (Bash with `run_in_background`), with `--no-browser`. Give it what the user wants to see:
   - a TorchRef result folder: `moorlight --no-browser path/to/run/` (loads refined.pdb + refined.mtz, or D.pdb + L_refined.pdb +
     difference_data.mtz)
   - files: `moorlight --no-browser model.pdb maps.mtz ligand.cif`; `file.mtz:label` names the maps of that file
   - `--port N` picks a port (default 8765). If it is busy, the next free port is used.
2. Wait for the line **`Moorlight ready: http://127.0.0.1:<port>/`** in the output and take the URL from it. Do not guess the port.
3. Open that URL in the **built-in browser pane** (the Claude Browser tools, e.g. `preview_start` with `url`, or `navigate` in an
   open pane). The viewer only renders, contours maps and answers `screenshot` while a browser tab is connected.
4. To check it worked, run `moorlight cmd state compact=true`: it lists the molecules and maps.

When the user already has a viewer running, reuse it: `~/.moorlight/server.json` holds the URL of the last one started, and
`moorlight cmd` reports an error if that viewer is gone. Several viewers can run at once on different ports; pass `--url` (or set
`MOORLIGHT_URL`) to address a specific one.

## 3. Drive it

Use the `mcp__moorlight__*` tools if this session has them (`claude mcp add moorlight -- python -m moorlight.mcp`, for future
sessions). Otherwise use the same commands through Bash, which always works:

```bash
moorlight cmd --list                          # every command with its parameters
moorlight cmd --list frame_atoms              # one command
moorlight cmd state compact=true
moorlight cmd goto chain=A resseq=85
moorlight cmd frame_atoms atoms='["A/85 SG", "B/501 C1"]'
moorlight cmd density_at_atoms chain=A start=84 end=86 maps=[1,3]
moorlight cmd screenshot width=800            # prints the PNG path: Read it to look at the view
```

Values are read as JSON when possible (numbers, `true`, lists), otherwise as strings; results are JSON. Atom specs:
`A/85 CA`, `A/85 LYS CA:B` (altloc B), `2#A/85 SG` (molecule 2), `A/85` (residue centroid).

What the commands cover:
- **Loading:** `load_model`, `load_mtz` (TorchRef, phenix, Fo-Fo and polder columns detected; `label`, `columns`, `d_min`),
  `load_map`, `load_torchref`, `reload` after a refinement finished.
- **Showing:** `goto`, `goto_atom`, `frame_atoms`, `set_center`, `set_map` (visible, level in σ, colour), `rename_map`,
  `set_molecule` / `solo` (visibility, carbon colour), `label_atoms`, `show_distance`, `screenshot`.
- **Measuring:** `density_at_atoms`, `density_profile`, `density_at_point`, `distance`, `peaks` (with whole-cell rank),
  `map_correlation`, `ligands`.
- **Building and refinement:** `refine` (map, difference-map or both targets), `accept_refinement` / `reject_refinement`,
  `fix_atoms`, `add_altconf`, `rotamers` / `set_rotamer`, `mutate`, `flip_peptide`, `add_water`, `add_ion`, `find_waters`,
  `add_ligand`, `add_residue`, `renumber`, `delete`, `set_occupancy`, `set_bfactor`, `undo` / `redo`, `save_model`.
- **Views and tours:** `save_view` / `restore_view`; `tour_add`, `tour_save`, `tour_load`, `tour_goto`. A saved tour can be stepped
  through with ◀ ▶ in the viewer without an agent.

## Working with the user

- **The user sees the same view.** Every command changes what is on screen, so say what you are showing. Do not hide the user's maps
  or molecules without saying so.
- **Measure, then describe.** Back statements with numbers from `density_at_atoms` / `density_profile` (σ of each map's whole-cell
  rms). Use a screenshot only as a sanity check.
- **Edits change the model in memory only.** `save_model` writes it; ask before overwriting a file.
- **Refinements stay open** until accepted or rejected; while one is open, other edits are refused.
- For guided walkthroughs of results, follow the **moorlight-result-tour** skill.
- The user's own Moorlight chat tab (Claude tab in the viewer) is a separate Claude session; it does not see this conversation.
