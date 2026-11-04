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
2. **After any restart of the viewer, re-establish the state.** A browser tab can keep showing the previous view and ignore the first commands
   (a view that looks wrong right after a restart may be stale: in one session an orientation looked wrong right after a restart and was fine in a fresh
   state, but the cause was never confirmed). Re-send `set_map`, `set_active_molecule` and `goto`, wait a moment, take a
   fresh screenshot, and only then judge the orientation. Use `roll` to test candidate rotations on screenshots instead of reasoning about conventions.
3. **Know what is loaded.** From `state`: which molecules (dark / light / waters), which maps (ids, difference vs 2mFo-DFc, level, visible).
   Never assume ids; read them. If maps or models for the next stop are missing, load them (`load_model`, `load_map`) before going there.
4. **Plan the stops from the analysis, not from the picture.** Take the list from the result tables (significant residues, tiered waters or other sites, strongest
   peaks via `peaks`). **Order the stops from the regions with the most significant changes to those with the least significant ones**, using the
   analysis's own significance measure (e.g. robust z of the residue, |σ| and tier of the water, strongest peak of the region) and nothing else.
   Group neighbouring items into one region (e.g. a ligand pocket) and rank regions by their strongest member. Say which measure you ranked by.
   When two data sets are compared, rank by the stronger evidence of the two and mark items found in only one of them. Tell the user the plan in
   3-6 lines before starting.

## For every stop

1. **Set up the display, one thing at a time.**
   - Show **one map at a time** when comparing two (e.g. two pipelines): `set_map` visible true for one, false for the other. Contour difference maps at
     the analysis threshold (±3σ unless the analysis used another value) and say which map is shown.
   - Frame the target with **`goto`** (defaults tuned to the user's own framing): it centres on the *side chain* (not the C-alpha), zooms so the side chain
     plus ~4.5 Å of context fits the screen (framed radius capped at ~9 Å for large residues) and keeps a thin depth slab (±2.8 Å) so only the relevant atoms are drawn. Override with
     `radius` / `slab` when the user asks for more or less; `radius=0` only re-centres. For waters and ions (single atoms) `goto` widens the frame and slab to
     include the hydrogen-bond / coordination partners (slab = farthest polar partner + 0.6 Å, frame = partner distance + ~4 Å). Set the map
     contour radius (`set_map radius`) to about the framed radius so the mesh does not fill the screen.
   - **Orientation:** `goto` also rotates the view so the long axis of the side chain is horizontal, the C-alpha on the left and the flat face towards the
     viewer (for 2-atom side chains the CB-SG type bond is the axis). `roll` (degrees about the viewing axis) turns it further, `orient=false` keeps the current
     rotation. Single atoms are not rotated.
   - **These defaults are one user's preferences** (fitted to the framing of a dozen residues, waters and a ligand). Treat them as a starting point:
     if the user changes the view, read it back with `state` (centre, zoom and slab are reported) and adjust `radius`, `slab` and `roll` for the next stops.
   - Waters and other atoms that are not in the active molecule: make the molecule that contains them active (`set_active_molecule`) before `goto` or
     `density_at_atoms`.
2. **Measure before you describe.** `density_at_atoms` (values in σ) works on the **active molecule** and only for the maps that are **visible**; to
   sample a water or another molecule make it active, and to compare two maps make each visible in turn. For positions that belong to no loaded
   atom, sample the map file directly at the coordinates (e.g. with gemmi) instead. For models with alternative conformers the
   keys are `atom:altloc`; compare conformers: density lost at conformer A and gained at conformer B means a real motion, a change at both or
   at neither does not. Backbone near zero and a strong side-chain pair means a side-chain motion (rotamer change), not noise.
3. **Take a screenshot and look at it**, but state only what the numbers support. A screenshot is a sanity check (is the density where I think it is,
   is the view cluttered), never the evidence. Say "the sulfur sits on …" only if the sampled values say so; if you cannot identify an atom in the
   image, do not name it.
4. **Report in a few lines:** a small table (atoms × maps/conformers, σ), what it means chemically, how the two data sets compare, and what is
   uncertain. State the sign convention of the map you show (for light-minus-dark maps: green = density gained in the light state, red = density lost).
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
