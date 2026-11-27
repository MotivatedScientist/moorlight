# Changelog

## 0.4.0 (unreleased)

- **Publication figures:** *Figure…* button and `figure` command: the current view at journal size (89/120/183 mm or
  16:9; 300/600 dpi), orthographic, maps as nested transparent surfaces (1, 1.5, 2.25 × level; weaker more transparent),
  carved to the shown atoms with an uncarved companion and a warning when carving hides density, labels as editable SVG
  text, settings JSON and a draft legend. `figure_tour` makes one figure per tour stop; `pymol=true` writes a PyMOL
  script with the same view, levels and carving. Colour-blind-safe palette (`palette=cvd`).
- Figure labels placed automatically off the atoms (avoiding labels, atoms and edges), paper-style text, leader lines.
- `cartoon` molecule style (main chain as a tube, side chains from Cα); metal coordination drawn as dashes.
- `show_distance style=hbond` and `show_hbonds`; `show_zone` draws only residues near given atoms.
- Lighter density surfaces in figures (opacities 0.12, 0.2, 0.32: at most 0.52 where all three overlap); `opacity`
  sets them per figure. Difference maps keep 0.22, 0.38, 0.6.
- Figure folder: *Folder* in the Figure dialog or `set_figure_dir`; remembered for the data, used by every figure and
  tour export (a subfolder per tour).
- *Preview* toggle (`figure_preview`): the viewer as the figure will look; maps stay mesh until the render.
- Figures without scripting: *figure look* (default on: cartoon main chain, residues within 5 Å of the labelled atoms, for
  the render only); dialog fields for levels and legend text; *Export figures…* in the tour box; colour squares open a
  colour picker; *Show only near*, *Show all* and *H-bonds* buttons for the clicked atoms. Clicked labels and
  Distance-tool lines are kept in saved views and tour stops.

## 0.3.0

### Shareable pages (`moorlight share`, by @HatPdotS, PR #4)
- One self-contained HTML viewer page with model, maps and tour stops; nothing is fetched (three.js 0.170.0 is vendored
  into the page). Runs on a phone, as a file, or as a claude.ai artifact (`--standalone` writes a full document; the default
  is an artifact fragment).
- Input from files, a TorchRef or phenix refinement folder, or a PDB id (PDB-REDO entry). Maps on their own grid
  (native, block or resampled box), limited to a region with `--around`.
- Mesh and Olex2-style slice modes, saved and preset planes (`--plane`, through atoms, copies at offsets), level sliders
  per map with rms shown, depth bar, hide residues in front, touch gestures.
- `.spec.json` and `.check.txt` next to every page; `moorlight share again page.spec.json …` rebuilds it with changes.
- `moorlight-share` skill for agents (also in the Claude Code plugin).

Review fixes before the merge:
- No local paths in pages (the command line stays in the `.spec.json` only); title, description and model names are
  escaped; pages declare UTF-8; non-finite values no longer blank a page.
- Coordination and planes use one conformer per residue; automatic planes stay near `--center` and inside `--around`,
  with a warning for requested planes outside it.
- A note in the summary when maps are resampled coarser than their grid (use `--around` for exact peak values).
- PDB-REDO downloads are announced and capped at 500 MB; `again` writes the page next to its spec only.

After the merge (from testing on real data):
- The "Unpacking maps…" overlay no longer stays over the view and blocks input.
- MTZ columns with labels that do not say the kind (e.g. `FDDCAT`) are classified from the map values;
  `file.mtz=name:F,PHI,diff`, `--diff MAP` / `--density MAP` set it; `again` keeps the kinds of the spec.
- `--slice` and `--open-plane NAME` open the page in Slice mode or on a plane.

### Viewer and agent API
- **Map slices** as in shared pages (*Maps as: Slice*): live plane facing the viewer, Fix plane, Depth, Thickness,
  plane through selected atoms; `slice` command for agents, with the strongest values on the plane. Kept in saved views.
- **Level sliders** per map as in shared pages (log scale up to the map's largest value, level in σ and map units).
- Maps whose labels do not tell the kind are classified from their values when first computed.
- TorchRef difference MTZ (new layout: `dF`, `FEXT`, `FWT` extrapolated map) detected with the right map names; a
  refinement folder with `*_dark.pdb` / `*_light.pdb` loads both models (light active, dark as reference) with the
  difference and extrapolated maps first.
- Saving keeps LINK records to atoms that exist only as alternate conformers (gemmi wrote them without atom name and
  distance, so refinement lost e.g. metal–carbamate restraints). Real-space refinement passes those links per conformer.
- A viewer tab opened while the browser pane is hidden sizes itself when shown; `screenshot` reports a clear error
  instead of an empty or 300×150 image.

### Claude Code plugin
- `claude-plugin/` with the `moorlight`, `moorlight-result-tour` and `moorlight-share` skills; the viewer opens in the
  session's built-in browser.

## 0.2.1
- Agent API: `moorlight cmd` CLI, display, framing and measurement commands, saved views and tours (PR #3); view radius.
