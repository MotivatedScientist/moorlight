# Changelog

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

### Viewer and agent API
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
