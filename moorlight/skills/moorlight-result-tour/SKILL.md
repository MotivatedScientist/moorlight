---
name: moorlight-result-tour
description: How to walk a user through crystallographic results (difference maps, moving residues, waters, comparisons between two data sets or pipelines) in a running Moorlight viewer. Use when the user asks for a tour, walkthrough, "show me", or to present/compare results with the mcp__moorlight__* tools.
---

# Guided result tours in Moorlight

A tour is a short sequence of **stops**. Each stop shows one residue, water or site, backs the picture with numbers, and ends
with one clear statement. The user drives the pace.

## Before the first stop

1. **Check the viewer is live.** Call `state`. If `screenshot` times out, no browser tab is connected: ask the user to open the viewer URL
   (from `~/.moorlight/server.json`) and do not continue silently.
2. **Know what is loaded.** From `state`: which molecules (dark / light / waters), which maps (ids, difference vs 2mFo-DFc, level, visible).
   Never assume ids; read them. If maps or models for the next stop are missing, load them (`load_model`, `load_map`) before going there.
3. **Plan the stops from the analysis, not from the picture.** Take the list from the result tables (significant residues, tier A/B waters, strongest
   peaks via `peaks`). Order them by story, e.g. the chromophore first, then the residues that respond to it, then waters, then time-dependence.
   Tell the user the plan in 3-6 lines before starting.

## For every stop

1. **Set up the display, one thing at a time.**
   - Show **one map at a time** when comparing two (e.g. two pipelines): `set_map` visible true for one, false for the other. Contour difference maps at
     the analysis threshold (±3σ unless the analysis used another value) and say which map is shown.
   - Frame the target with **`goto` (default `radius` 5 Å)**: it centres on the residue, zooms so the whole residue plus ~5 Å fits the screen and keeps a
     depth slab around it so only the relevant atoms are drawn. Use a smaller `slab` for a cleaner view, `radius=0` to only re-centre. Set the map
     contour radius (`set_map radius`) to about the framed radius so the mesh does not fill the screen.
   - Waters and other atoms that are not in the active molecule: make the molecule that contains them active (`set_active_molecule`) before `goto` or
     `density_at_atoms`.
2. **Measure before you describe.** `density_at_atoms` (values in σ, for every *visible* map) on the residue. For models with alternative conformers the
   keys are `atom:altloc`; compare conformers: density lost at conformer A and gained at conformer B means a real motion, a change at both or
   at neither does not. Backbone near zero and a strong side-chain pair means a side-chain motion (rotamer change), not noise.
3. **Take a screenshot and look at it**, but state only what the numbers support. A screenshot is a sanity check (is the density where I think it is,
   is the view cluttered), never the evidence. Say "the sulfur sits on …" only if the sampled values say so; if you cannot identify an atom in the
   image, do not name it.
4. **Report in a few lines:** a small table (atoms × maps/conformers, σ), what it means chemically, how the two data sets compare, and what is
   uncertain. Use the convention green = density gained, red = density lost (light minus dark).
5. **Stop and ask** whether to continue. Offer the next stop by name. Do not run the whole tour unprompted.

## Things to be careful about

- **Do not trust residue classifiers blindly.** Labels such as "shift-like" / "disorder-like" come from rules that can average over conformers. If the map
  contradicts the label (e.g. strong positive density on the second conformer), say so and let the user judge.
- **Peak heights in σ of whole-cell maps are not comparable between data sets** when the noise differs (the map rms contains noise and signal).
  When comparing two pipelines, prefer half-map based measures (cross-half CC, signal against half-map noise); if you quote raw σ values, say that.
- **Significance tiers** (A/B/C for waters, tier for residues) belong to the analysis that produced them; quote them with the criteria, do not re-derive.
- **Say what you did not look at** (time points, chains, maps not loaded).
- **Clutter:** the API cannot hide a molecule. If several models overlap, tell the user which to hide in the panel, or use a thin slab.
- Keep messages short; the user often reads them next to the 3D view. One stop = one message.
