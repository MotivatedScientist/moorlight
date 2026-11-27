// Moorlight – browser viewer. Coot 0.9-style controls on top of three.js.
import * as THREE from "three";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { marked } from "marked";

const $ = (s) => document.querySelector(s);
const ELEMENT_COLORS = { C: 0xe6c229, N: 0x3060ff, O: 0xff2020, S: 0xe0e000, P: 0xff8000, H: 0xdddddd,
  SE: 0xe0a000, FE: 0xe07000, MG: 0x20e020, ZN: 0x8080c0, CA: 0x40c040, NA: 0x8040ff, CL: 0x20f020, K: 0x8040c0, MN: 0x9c7ac7 };
const CARBON_BY_MOL = [0xe6c229, 0x60c0ff, 0x80e080, 0xff80c0, 0xc0a0ff, 0xffffff];
const INTERMEDIATE = 0xf0f0f0;

// ------------------------------------------------------------------ state
const S = {
  mols: new Map(),       // id -> {data, group, residues, xyz}
  maps: new Map(),       // id -> {info, obj(Group), lastCenter, lastLevel}
  activeMol: null,
  scrollMap: null,
  refinementMap: null,
  center: new THREE.Vector3(),
  zoom: 30, slab: 9,
  labels: new Map(),     // "mol:idx" -> div
  distances: [],
  selection: [],         // [{mol, idx}]
  clicked: null,         // {mol, idx}
  tool: null, toolClicks: [],
  rsr: null,             // {mol, idx:[], xyz:Float32Array|null}
  cross: true,
};

// ------------------------------------------------------------------ three
const vp = $("#viewport");
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(window.devicePixelRatio);
vp.prepend(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);
scene.fog = new THREE.Fog(0x000000, 20, 40);
const camera = new THREE.PerspectiveCamera(30, 1, 1, 1000);
const rot = new THREE.Group();     // rotation about the centre
const world = new THREE.Group();   // translation (-centre)
rot.add(world); scene.add(rot);
const overlay = new THREE.Group(); world.add(overlay);
scene.add(new THREE.AmbientLight(0xffffff, 1.1));
const keyLight = new THREE.DirectionalLight(0xffffff, 2.0);
keyLight.position.set(0.4, 0.6, 1);  // fixed relative to the viewer (the model rotates, not the camera)
scene.add(keyLight);
const lineMaterials = new Set();

function lineMat(width, opts = {}) {
  const m = new LineMaterial({ linewidth: width, vertexColors: true, worldUnits: false, ...opts });
  lineMaterials.add(m);
  return m;
}

// A tab opened while the browser pane is hidden has a 0×0 viewport: keep the last size (or none yet) and size the
// canvas once it is shown, instead of setting a NaN aspect that leaves the view empty.
let sized = false;   // the canvas has had a real size at least once
function resize() {
  const w = vp.clientWidth, h = vp.clientHeight;
  if (!w || !h) return false;
  sized = true;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  for (const m of lineMaterials) m.resolution.set(w * devicePixelRatio, h * devicePixelRatio);
  return true;
}
window.addEventListener("resize", resize);
document.addEventListener("visibilitychange", () => { if (!document.hidden && resize()) { redraw(); scheduleContour(true); } });

function updateCamera() {
  camera.position.set(0, 0, S.zoom);
  camera.near = Math.max(0.5, S.zoom - S.slab);
  camera.far = S.zoom + S.slab;
  camera.updateProjectionMatrix();
  scene.fog.near = S.zoom - S.slab * 0.3;
  scene.fog.far = S.zoom + S.slab * 1.05;
  world.position.copy(S.center).multiplyScalar(-1);
}

// centre cross
const crossObj = (() => {
  const g = new THREE.BufferGeometry();
  const s = 0.5;
  g.setAttribute("position", new THREE.Float32BufferAttribute([-s,0,0, s,0,0, 0,-s,0, 0,s,0, 0,0,-s, 0,0,s], 3));
  const o = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0xbbbbbb, fog: false }));
  rot.add(o);
  return o;
})();

let needsRender = true;
function render() {
  requestAnimationFrame(render);
  animateCenter();
  if (!needsRender) return;
  needsRender = false;
  updateCamera();
  updateSlice();
  crossObj.visible = S.cross && !S.preview;
  renderer.render(scene, camera);
  updateLabels();
  if (S.preview) updatePreviewLabels();
}
const redraw = () => (needsRender = true);

// ------------------------------------------------------------- networking
let ws;
function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onmessage = (ev) => onServer(JSON.parse(ev.data));
  ws.onclose = () => { status("connection lost – retrying…"); setTimeout(connect, 1500); };
  ws.onopen = () => status("connected");
}
const send = (m) => ws && ws.readyState === 1 && ws.send(JSON.stringify(m));

async function cmd(name, args = {}) {
  const r = await fetch("/api/cmd", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cmd: name, args, source: "ui" }) }).then((r) => r.json());
  if (!r.ok) { status("⚠ " + r.error); throw new Error(r.error); }
  return r.result;
}

let viewTimer = null;
function pushView() {
  clearTimeout(viewTimer);
  viewTimer = setTimeout(() => {
    const maps = {};
    for (const [id, m] of S.maps) maps[id] = { visible: m.info.visible, level_sigma: m.info.level_sigma };
    const sel = S.selection.filter((s) => s.mol === S.activeMol).map((s) => s.idx);
    const c = anim ? anim.to : S.center;   // while animating, report where the view is going
    send({ type: "view", center: c.toArray().map((v) => +v.toFixed(2)), zoom: +S.zoom.toFixed(2), slab: +S.slab.toFixed(2),
      orient: rot.quaternion.toArray().map((v) => +v.toFixed(5)),
      selection: sel, clicked: S.clicked ? atomLabel(S.clicked.mol, S.clicked.idx) : null,
      labelled: [...S.labels.keys()].map((k) => k.split(":").map(Number)),
      active_map: S.scrollMap, maps });
  }, 150);
}

function onServer(m) {
  switch (m.type) {
    case "model": setMolecule(m.mol); renderRsrSelects(); ligCache = null; break;
    case "close_molecule": removeMolecule(m.mol); break;
    case "active_molecule": S.activeMol = m.mol; renderMolList(); break;
    case "maps": setMaps(m.maps, m.refinement_map); renderRsrSelects(); break;
    case "rsr_settings": setRsrSettings(m.settings); break;
    case "maps_changed": for (const mm of S.maps.values()) mm.lastCenter = null; SLICE.box = null; scheduleContour(true); break;
    case "view_cmd":
      if (m.center) moveTo(new THREE.Vector3(...m.center));
      if (m.orient) { rot.quaternion.set(...m.orient); needsRender = true; }   // long axis of the side chain horizontal, flat face to the viewer
      if (m.fit_radius) {   // frame a sphere of fit_radius Å: distance so that it fits the smaller screen dimension; slab = depth half-thickness
        const half = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * Math.min(1, camera.aspect);
        S.zoom = Math.min(400, Math.max(4, m.fit_radius / half));
        S.slab = Math.min(200, Math.max(1, m.slab || m.fit_radius));
      } else {              // restored view: zoom and slab as saved
        if (m.zoom) S.zoom = m.zoom;
        if (m.slab) S.slab = m.slab;
      }
      redraw(); scheduleContour(true); pushView();
      break;
    case "mol_display": applyDisplay(m.mol, m); break;
    case "slice": applySlice(m); break;
    case "figure_preview": setPreview(m.on, m.radius); break;
    case "annotate": annotate(m); break;
    case "tour": renderTour(m); break;
    case "log": addLog(m.text); break;
    case "rsr_start": startRSR(m); renderUndo(); break;
    case "rsr_coords": rsrCoords(m); break;
    case "rsr_status": rsrStatus(m); break;
    case "rsr_end": endRSR(); renderUndo(); break;
    case "view_request":
      send({ type: "reply", id: m.id, data: { center: (anim ? anim.to : S.center).toArray(), orient: rot.quaternion.toArray(),
        zoom: S.zoom, slab: S.slab, aspect: camera.aspect || 1 } });
      break;
    case "figure_request":
      renderFigure(m).then((data) => send({ type: "reply", id: m.id, data }),
        (err) => { console.error(err); send({ type: "reply", id: m.id, data: `error: ${err}` }); });
      break;
    case "screenshot_request":
      takeScreenshot(m).then((data) => send({ type: "reply", id: m.id, data }), (err) => send({ type: "reply", id: m.id, data: `error: ${err}` }));
      break;
    case "remote_cmd": status(`Claude: ${m.cmd} ${JSON.stringify(m.args).slice(0, 120)}`); break;
    case "chat": chatAdd(m); break;
    case "chat_delta_start": chatStreamStart(); break;
    case "chat_delta": chatStreamDelta(m.text); break;
    case "chat_busy": chatBusy(m.busy); break;
    case "chat_done": $("#chat-status").textContent = `${m.seconds}s${m.cost ? ` · $${m.cost.toFixed(3)}` : ""}`; break;
    case "chat_settings": $("#chat-bash").checked = m.settings.allow_bash; break;
    case "chat_reset": $("#chat-log").innerHTML = ""; break;
  }
}
function render1() { updateCamera(); updateSlice(); renderer.render(scene, camera); }

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
// Screenshot after the view has settled: centre animation done and every visible map contoured for this centre/level.
async function takeScreenshot(opt = {}) {
  if (opt.wait !== false) {
    for (let t = 0; anim && t < 100; t++) await sleep(20);
    for (let t = 0; t < 600; t++) {
      clearTimeout(contourTimer);
      while (contourBusy) await sleep(30);
      await contourAll();
      const stale = (SLICE.style !== "slice" && [...S.maps.values()].some((m) => m.info.visible &&
        !(m.lastCenter && m.lastCenter.distanceTo(S.center) < 0.5 && m.lastLevel === m.info.level_sigma))) ||
        (SLICE.style === "slice" && !(await sliceReady()));
      if (!stale && !anim) break;
      await sleep(50);
    }
  }
  resize();
  const src = renderer.domElement;
  if (!sized)
    throw new Error("the viewer tab is hidden or has zero size (it was opened while the browser pane was hidden); show it and retry");
  needsRender = false;
  render1(); updateLabels();
  const W = src.width, H = src.height, k = src.clientWidth ? W / src.clientWidth : devicePixelRatio;
  const cv = document.createElement("canvas");
  cv.width = W; cv.height = H;
  const g = cv.getContext("2d");
  g.drawImage(src, 0, 0);
  if (opt.labels !== false && S.preview) {      // figure preview: its placed labels
    const half_w = S.zoom * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * camera.aspect;
    const labs = figureLabels(camera, src.clientWidth, src.clientHeight, 13, half_w);
    g.save(); g.scale(k, k); drawLabels(g, labs, 13, "white"); g.restore();
  } else if (opt.labels !== false) {   // the labels are HTML on top of the canvas: draw them in
    g.font = `${12 * k}px ui-monospace, Menlo, monospace`;
    g.textBaseline = "middle";
    g.shadowColor = "#000"; g.shadowBlur = 3 * k;
    const drawDiv = (div, color) => {
      if (div.style.display === "none" || div.style.visibility === "hidden") return;
      g.fillStyle = color;
      g.fillText(div.textContent, parseFloat(div.style.left) * k + 6 * k, parseFloat(div.style.top) * k);
    };
    for (const div of S.labels.values()) drawDiv(div, "#ffffff");
    for (const d of S.distances) drawDiv(d.div, "#99ffff");
  }
  let [cx, cy, cw, ch] = [0, 0, W, H];
  if (Array.isArray(opt.crop) && opt.crop.length === 4) {
    const [fx, fy, fw, fh] = opt.crop.map(Number);
    cx = Math.round(fx * W); cy = Math.round(fy * H); cw = Math.round(fw * W); ch = Math.round(fh * H);
  }
  const outW = Math.round(opt.width || Math.min(cw, 1000)), outH = Math.round(ch * outW / cw);
  const out = document.createElement("canvas");
  out.width = outW; out.height = outH;
  const go = out.getContext("2d");
  go.imageSmoothingQuality = "high";
  go.drawImage(cv, cx, cy, cw, ch, 0, 0, outW, outH);
  return out.toDataURL("image/png");
}

// --------------------------------------------------------------- molecules
function carbonColor(molId) {
  const own = S.mols.get(molId)?.color;
  if (own) return new THREE.Color(own).getHex();
  const ids = [...S.mols.keys()].sort((a, b) => a - b);
  const k = Math.max(0, ids.indexOf(molId));
  return CARBON_BY_MOL[k % CARBON_BY_MOL.length];
}
function elColor(el, molId) {
  el = (el || "").toUpperCase();
  if (el === "C") return carbonColor(molId);
  return ELEMENT_COLORS[el] ?? 0xff40ff;
}

function setMolecule(data) {
  let mol = S.mols.get(data.id);
  if (!mol) {
    mol = { group: new THREE.Group(), hideH: true, style: "sticks" };
    world.add(mol.group);
    S.mols.set(data.id, mol);
    if (S.activeMol == null) S.activeMol = data.id;
  }
  mol.data = data;
  mol.xyz = Float32Array.from(data.xyz);
  mol.fixed = new Set(data.fixed);
  mol.residues = buildResidues(data);
  setDisplayFields(mol, data.display ?? { visible: data.visible });
  rebuildMolGeometry(data.id);
  // drop labels pointing at atoms that no longer exist
  for (const [k, div] of S.labels) {
    const [mid, idx] = k.split(":").map(Number);
    if (mid === data.id && idx >= data.atoms.length) { div.remove(); S.labels.delete(k); }
  }
  renderMolList();
  redraw();
}

// display settings owned by the server (set_molecule), so `state` reports what the user sees
function setDisplayFields(mol, d) {
  mol.group.visible = d.visible !== false;
  mol.color = d.color || null;
  if ("style" in d) mol.style = d.style || "sticks";   // null = the default style (also after a figure look is undone)
  if (d.hydrogens != null) mol.hideH = !d.hydrogens;
  if ("zone" in d) mol.zone = d.zone ? new Set(d.zone) : null;
}
function applyDisplay(id, d) {
  const mol = S.mols.get(id);
  if (!mol) return;
  const before = `${mol.color}|${mol.style}|${mol.hideH}|${mol.zone ? [...mol.zone].join(",") : ""}`;
  setDisplayFields(mol, d);
  if (`${mol.color}|${mol.style}|${mol.hideH}|${mol.zone ? [...mol.zone].join(",") : ""}` !== before) rebuildMolGeometry(id);
  renderMolList(); redraw();
}

function renderUndo() {
  const d = S.mols.get(S.activeMol)?.data;
  const u = d?.undo ?? 0, r = d?.redo ?? 0;
  $("#btn-undo").textContent = u ? `↶ Undo (${u})` : "↶ Undo";
  $("#btn-redo").textContent = r ? `↷ Redo (${r})` : "↷ Redo";
  $("#btn-undo").disabled = !u && !S.rsr;
  $("#btn-redo").disabled = !r;
}

function removeMolecule(id) {
  const mol = S.mols.get(id);
  if (!mol) return;
  world.remove(mol.group);
  S.mols.delete(id);
  if (S.activeMol === id) S.activeMol = S.mols.keys().next().value ?? null;
  renderMolList(); redraw();
}

function buildResidues(data) {
  const res = [];
  let last = null;
  data.atoms.forEach((a, i) => {
    const key = `${a[0]}/${a[1]}${a[2]}`;
    if (!last || last.key !== key) { last = { key, chain: a[0], num: a[1], icode: a[2], name: a[3], atoms: [], ca: i }; res.push(last); }
    last.atoms.push(i);
    if (a[4] === "CA") last.ca = i;
  });
  return res;
}

function bondGeometry(xyz, bonds, elements, molId, include = null, color = null) {
  const pos = [], col = [];
  const c = new THREE.Color();
  for (let b = 0; b < bonds.length; b += 2) {
    const i = bonds[b], j = bonds[b + 1];
    if (include && !(include.has(i) || include.has(j))) continue;
    const xi = xyz[3*i], yi = xyz[3*i+1], zi = xyz[3*i+2];
    const xj = xyz[3*j], yj = xyz[3*j+1], zj = xyz[3*j+2];
    const mx = (xi + xj) / 2, my = (yi + yj) / 2, mz = (zi + zj) / 2;
    pos.push(xi, yi, zi, mx, my, mz, mx, my, mz, xj, yj, zj);
    c.setHex(color ?? elColor(elements[i], molId)); col.push(c.r, c.g, c.b, c.r, c.g, c.b);
    c.setHex(color ?? elColor(elements[j], molId)); col.push(c.r, c.g, c.b, c.r, c.g, c.b);
  }
  return { pos, col };
}

function crossGeometry(xyz, indices, elements, molId, size) {
  const pos = [], col = [];
  const c = new THREE.Color();
  for (const i of indices) {
    const x = xyz[3*i], y = xyz[3*i+1], z = xyz[3*i+2];
    const s = size(i);
    pos.push(x-s,y,z, x+s,y,z, x,y-s,z, x,y+s,z, x,y,z-s, x,y,z+s);
    c.setHex(elColor(elements[i], molId));
    for (let k = 0; k < 6; k++) col.push(c.r, c.g, c.b);
  }
  return { pos, col };
}

function makeLines(pos, col, width) {
  const g = new LineSegmentsGeometry();
  if (pos.length) { g.setPositions(pos); g.setColors(col); }
  const o = new LineSegments2(g, lineMat(width));
  o.material.resolution.set(vp.clientWidth * devicePixelRatio, vp.clientHeight * devicePixelRatio);
  return o;
}

function rebuildMolGeometry(id) {
  const mol = S.mols.get(id);
  const d = mol.data;
  for (const ch of [...mol.group.children]) { mol.group.remove(ch); ch.traverse((o) => o.geometry?.dispose()); }
  const isH = (i) => d.element[i] === "H";
  const out = (i) => mol.zone && !mol.zone.has(i);     // outside the shown zone (show_zone)
  // cartoon: the main chain as a tube through CA, side chains from CA as sticks
  const cartoon = mol.style === "cartoon" ? cartoonResidues(mol) : null;
  const hiddenBB = (i) => cartoon && cartoon.bb.has(i);
  const isMetal = (i) => METALS.has((d.element[i] || "").toUpperCase());
  const keep = [];  // bond indices to draw
  const coord = []; // metal-ligand bonds: drawn dashed
  for (let k = 0; k < d.bonds.length / 2; k++) {
    const i = d.bonds[2*k], j = d.bonds[2*k+1];
    if (mol.hideH && (isH(i) || isH(j))) continue;
    if (out(i) || out(j)) continue;
    if (isMetal(i) !== isMetal(j)) { coord.push(i, j); continue; }
    if (hiddenBB(i) || hiddenBB(j)) continue;
    keep.push(k);
  }
  const bonded = new Uint8Array(d.atoms.length);
  for (let k = 0; k < d.bonds.length / 2; k++) {
    const i = d.bonds[2*k], j = d.bonds[2*k+1];
    if (isMetal(i) === isMetal(j)) { bonded[i] = 1; bonded[j] = 1; }
  }
  const lone = [];
  for (let i = 0; i < d.atoms.length; i++) if (!bonded[i] && !(mol.hideH && isH(i)) && !out(i)) lone.push(i);
  if (mol.style === "lines") {
    const bonds = [];
    for (const k of keep) bonds.push(d.bonds[2*k], d.bonds[2*k+1]);
    const b = bondGeometry(mol.xyz, bonds, d.element, id);
    mol.group.add(makeLines(b.pos, b.col, 3));
    const isWater = (i) => ["HOH", "WAT", "DOD"].includes(d.atoms[i][3]);
    const cr = crossGeometry(mol.xyz, lone, d.element, id, (i) => (isWater(i) ? 0.3 : 0.55));
    if (cr.pos.length) mol.group.add(makeLines(cr.pos, cr.col, 2));
    if (coord.length) {
      const pos = [], col = [];
      for (let n = 0; n < coord.length; n += 2) for (const [a, b] of dashes(P3(mol, coord[n]), P3(mol, coord[n+1]))) {
        pos.push(a.x, a.y, a.z, b.x, b.y, b.z); col.push(0.6, 0.6, 0.6, 0.6, 0.6, 0.6);
      }
      mol.group.add(makeLines(pos, col, 2));
    }
  } else {
    for (const o of stickMeshes(mol, id, keep, lone, cartoon)) mol.group.add(o);
    if (coord.length) mol.group.add(dashedSticks(coord.map((i) => P3(mol, i)), COORD_COLOR, 0.055));
    mol.hasCoord = coord.length > 0;
    if (cartoon) mol.group.add(cartoonTube(mol, id, cartoon));
  }
  // fixed atoms: magenta crosses
  if (mol.fixed.size) {
    const pos = [], col = [];
    for (const i of mol.fixed) {
      const x = mol.xyz[3*i], y = mol.xyz[3*i+1], z = mol.xyz[3*i+2], s = 0.35;
      pos.push(x-s,y-s,z, x+s,y+s,z, x-s,y+s,z, x+s,y-s,z);
      for (let k = 0; k < 4; k++) col.push(1, 0.2, 1);
    }
    mol.group.add(makeLines(pos, col, 2));
  }
}

// ------------------------------------------------- cartoon, dashed bonds
const METALS = new Set(["LI", "NA", "K", "RB", "CS", "MG", "CA", "SR", "BA", "MN", "FE", "CO", "NI", "CU", "ZN", "CD", "HG",
  "PT", "PD", "AG", "AU", "MO", "W", "V", "CR", "AL", "GA"]);
const COORD_COLOR = 0x9a9a9a;
const BACKBONE = new Set(["N", "C", "O", "OXT", "H", "HA", "H1", "H2", "H3", "HA2", "HA3"]);
const P3 = (mol, i) => new THREE.Vector3(mol.xyz[3*i], mol.xyz[3*i+1], mol.xyz[3*i+2]);
// [start, end] pairs of the dashes between a and b
function dashes(a, b, dash = 0.2, gap = 0.14) {
  const len = a.distanceTo(b), dir = b.clone().sub(a).normalize(), out = [];
  for (let t = 0.05; t < len - 0.05; t += dash + gap) {
    const e = Math.min(t + dash, len - 0.05);
    out.push([a.clone().addScaledVector(dir, t), a.clone().addScaledVector(dir, e)]);
  }
  return out;
}
// dashed sticks for pairs of points [a0, b0, a1, b1, ...]
function dashedSticks(pts, color, r, dash, gap) {
  const segs = [];
  for (let n = 0; n < pts.length; n += 2) segs.push(...dashes(pts[n], pts[n+1], dash, gap));
  const im = new THREE.InstancedMesh(S.preview ? FIG.cyl : cylGeo, S.preview ? FIG.mat : solidMat, Math.max(1, segs.length));
  im.count = segs.length;
  segs.forEach(([p, q], n) => {
    _v.subVectors(q, p);
    const len = _v.length();
    _q.setFromUnitVectors(_up, _v.normalize());
    _m.compose(p.clone().add(q).multiplyScalar(0.5), _q, _s.set(r, len, r));
    im.setMatrixAt(n, _m);
    im.setColorAt(n, _c.setHex(color));
  });
  im.frustumCulled = false;
  return im;
}
// protein residues drawn as cartoon: their main-chain atoms are hidden (side chains start at CA). Within a zone the
// tube runs three residues past the zone on both sides so it reads as a chain.
function cartoonResidues(mol) {
  const d = mol.data, bb = new Set(), tube = [];
  const prot = mol.residues.filter((r) => r.atoms.some((i) => d.atoms[i][4] === "N") && r.atoms.some((i) => d.atoms[i][4] === "C")
    && d.atoms[r.ca][4] === "CA");
  const inZone = (r) => !mol.zone || r.atoms.some((i) => mol.zone.has(i));
  const shown = new Set();
  prot.forEach((r, k) => { if (inZone(r)) for (let j = Math.max(0, k - 3); j <= Math.min(prot.length - 1, k + 3); j++) shown.add(j); });
  prot.forEach((r, k) => {
    for (const i of r.atoms) if (BACKBONE.has(d.atoms[i][4])) bb.add(i);
    if (shown.has(k)) tube.push(r);
  });
  return { bb, tube, ca: new Set(prot.map((r) => r.ca)) };
}
function cartoonTube(mol, id, cartoon) {
  const grp = new THREE.Group();
  let seg = [], last = null;
  const flush = () => {
    if (seg.length >= 2) {
      const curve = new THREE.CatmullRomCurve3(seg, false, "centripetal");
      const g = new THREE.TubeGeometry(curve, Math.max(8, seg.length * 8), 0.24, 14, false);
      grp.add(new THREE.Mesh(g, cartoonMat(id)));
    }
    seg = [];
  };
  for (const r of cartoon.tube) {
    const p = P3(mol, r.ca);
    if (last && (last.r.chain !== r.chain || last.p.distanceTo(p) > 4.3)) flush();
    seg.push(p);
    last = { r, p };
  }
  flush();
  return grp;
}
const cartoonMats = new Map();
function cartoonMat(id) {
  const c = carbonColor(id);
  if (!cartoonMats.has(c)) cartoonMats.set(c, new THREE.MeshPhongMaterial({ color: new THREE.Color(c).lerp(new THREE.Color(0xffffff), 0.25),
    shininess: 30, specular: 0x222222, side: THREE.DoubleSide }));
  return cartoonMats.get(c);
}

// ---------------------------------------------------------- ball & stick
const STICK_R = 0.13, BALL_R = 0.24, H_STICK_R = 0.07, H_BALL_R = 0.12, WATER_R = 0.28, ION_R = 0.6;
const sphereGeo = new THREE.SphereGeometry(1, 12, 8);
const cylGeo = new THREE.CylinderGeometry(1, 1, 1, 10, 1, true);
const solidMat = new THREE.MeshLambertMaterial({ color: 0xffffff });
const _m = new THREE.Matrix4(), _q = new THREE.Quaternion(), _v = new THREE.Vector3(), _s = new THREE.Vector3();
const _up = new THREE.Vector3(0, 1, 0), _c = new THREE.Color();

function stickMeshes(mol, id, keep, lone, cartoon = null) {
  const d = mol.data, x = mol.xyz;
  const P = (i) => new THREE.Vector3(x[3*i], x[3*i+1], x[3*i+2]);
  const isH = (i) => d.element[i] === "H";
  const isWater = (i) => ["HOH", "WAT", "DOD"].includes(d.atoms[i][3]);
  // neighbours (heavy atoms) for placing the second line of double bonds
  const nb = new Map();
  for (const k of keep) {
    const i = d.bonds[2*k], j = d.bonds[2*k+1];
    if (isH(i) || isH(j)) continue;
    (nb.get(i) || nb.set(i, []).get(i)).push(j);
    (nb.get(j) || nb.set(j, []).get(j)).push(i);
  }
  const cyl = [];  // [p, q, r, color]
  for (const k of keep) {
    const i = d.bonds[2*k], j = d.bonds[2*k+1];
    const pi = P(i), pj = P(j), mid = pi.clone().add(pj).multiplyScalar(0.5);
    const r = isH(i) || isH(j) ? H_STICK_R : STICK_R;
    const ci = elColor(d.element[i], id), cj = elColor(d.element[j], id);
    cyl.push([pi, mid, r, ci], [mid, pj, r, cj]);
    const order = d.orders ? d.orders[k] : 1;
    if (order === 2 || order === 3 || order === 4) {
      // offset towards a neighbour (inside of rings), shortened at both ends, like Coot
      const axis = pj.clone().sub(pi);
      const c = (nb.get(i) || []).find((n) => n !== j) ?? (nb.get(j) || []).find((n) => n !== i);
      let off;
      if (c !== undefined) {
        const pc = P(c);
        const ref = (nb.get(i) || []).includes(c) ? pi : pj;
        off = pc.sub(ref);
        off.sub(axis.clone().multiplyScalar(off.dot(axis) / axis.lengthSq()));
      }
      if (!off || off.lengthSq() < 1e-6) off = new THREE.Vector3(1, 0, 0).cross(axis);
      if (off.lengthSq() < 1e-6) off = new THREE.Vector3(0, 1, 0).cross(axis);
      off.normalize().multiplyScalar(0.27);
      const a = pi.clone().lerp(pj, 0.15).add(off), b = pj.clone().lerp(pi, 0.15).add(off);
      const m2 = a.clone().add(b).multiplyScalar(0.5);
      cyl.push([a, m2, r * 0.6, ci], [m2, b, r * 0.6, cj]);
      if (order === 3) {
        const a3 = pi.clone().lerp(pj, 0.15).sub(off), b3 = pj.clone().lerp(pi, 0.15).sub(off);
        const m3 = a3.clone().add(b3).multiplyScalar(0.5);
        cyl.push([a3, m3, r * 0.6, ci], [m3, b3, r * 0.6, cj]);
      }
    }
  }
  const out = [];
  if (cyl.length) {
    const im = new THREE.InstancedMesh(S.preview ? FIG.cyl : cylGeo, S.preview ? FIG.mat : solidMat, cyl.length);
    cyl.forEach(([p, q, r, col], n) => {
      _v.subVectors(q, p);
      const len = _v.length();
      _q.setFromUnitVectors(_up, _v.normalize());
      _m.compose(p.clone().add(q).multiplyScalar(0.5), _q, _s.set(r, len, r));
      im.setMatrixAt(n, _m);
      im.setColorAt(n, _c.setHex(col));
    });
    im.frustumCulled = false;
    out.push(im);
  }
  const loneSet = new Set(lone);
  const allBalls = [];
  for (let i = 0; i < d.atoms.length; i++) {
    if ((mol.hideH && isH(i)) || (mol.zone && !mol.zone.has(i))) continue;
    if (cartoon && (cartoon.bb.has(i) || cartoon.ca.has(i))) continue;   // the tube carries the main chain
    allBalls.push(i);
  }
  if (allBalls.length) {
    const im = new THREE.InstancedMesh(S.preview ? FIG.sphere : sphereGeo, S.preview ? FIG.mat : solidMat, allBalls.length);
    allBalls.forEach((i, n) => {
      let r = isH(i) ? H_BALL_R : BALL_R;
      if (loneSet.has(i)) r = isWater(i) ? WATER_R : (isH(i) ? H_BALL_R : ION_R);
      _m.compose(P(i), _q.identity(), _s.set(r, r, r));
      im.setMatrixAt(n, _m);
      im.setColorAt(n, _c.setHex(elColor(d.element[i], id)));
    });
    im.frustumCulled = false;
    out.push(im);
  }
  return out;
}

function atomLabel(molId, i) {
  const a = S.mols.get(molId).data.atoms[i];
  return `${a[0]}/${a[1]}${a[2]} ${a[3]} ${a[4]}${a[5] ? ":" + a[5] : ""}`;
}
function atomInfo(molId, i) {
  const a = S.mols.get(molId).data.atoms[i];
  const x = S.mols.get(molId).xyz;
  return `${atomLabel(molId, i)}  occ ${a[6].toFixed(2)}  B ${a[7].toFixed(1)}  ` +
    `(${x[3*i].toFixed(2)}, ${x[3*i+1].toFixed(2)}, ${x[3*i+2].toFixed(2)})  mol ${molId}`;
}

// --------------------------------------------------------------------- maps
function setMaps(list, refinementMap) {
  S.refinementMap = refinementMap;
  const seen = new Set();
  for (const info of list) {
    seen.add(info.id);
    let m = S.maps.get(info.id);
    if (!m) {
      m = { obj: new THREE.Group(), lastCenter: null, lastLevel: null };
      world.add(m.obj);
      S.maps.set(info.id, m);
    }
    const changed = !m.info || m.info.level_sigma !== info.level_sigma || m.info.radius !== info.radius || m.info.color !== info.color;
    m.info = info;
    if (changed) m.lastCenter = null;
  }
  for (const [id, m] of S.maps) if (!seen.has(id)) {
    for (const ch of m.obj.children) ch.geometry.dispose();
    world.remove(m.obj); S.maps.delete(id); redraw();
  }
  if (S.scrollMap == null || !S.maps.has(S.scrollMap)) {
    const vis = [...S.maps.values()].filter((m) => m.info.visible);
    S.scrollMap = vis.length ? vis[0].info.id : (S.maps.keys().next().value ?? null);
  }
  applyMapVisibility();
  renderMapList();
  scheduleContour(true);
}
// mesh contours are drawn in Mesh style only (Slice style shows the map on the slice plane instead)
function applyMapVisibility() {
  for (const m of S.maps.values()) m.obj.visible = m.info.visible && SLICE.style !== "slice";
  redraw();
}

let contourTimer = null, contourBusy = false;
function scheduleContour(now = false) {
  clearTimeout(contourTimer);
  contourTimer = setTimeout(contourAll, now ? 0 : 120);
}
async function contourAll() {
  if (contourBusy) { scheduleContour(); return; }
  contourBusy = true;
  try {
    for (const [id, m] of S.maps) {
      if (!m.info.visible || SLICE.style === "slice") continue;
      if (m.lastCenter && m.lastCenter.distanceTo(S.center) < 0.5 && m.lastLevel === m.info.level_sigma) continue;
      await contourMap(id);
    }
  } finally { contourBusy = false; }
}
async function contourMap(id) {
  const m = S.maps.get(id);
  const c = S.center.clone();
  const level = m.info.level_sigma;
  const url = `/api/map/${id}/mesh?x=${c.x}&y=${c.y}&z=${c.z}&r=${m.info.radius}&level=${level}`;
  const resp = await fetch(url);
  if (!resp.ok) {   // e.g. a column that is not in the MTZ: report once, do not retry for this view
    status(`⚠ map ${id} (${m.info.name}): ${(await resp.text()).slice(0, 200)}`);
    m.lastCenter = c; m.lastLevel = level;
    return;
  }
  const buf = await resp.arrayBuffer();
  const hlen = new DataView(buf).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hlen)));
  if (header.max_sigma != null && m.info.max_sigma == null) { m.info.max_sigma = header.max_sigma; updateLevelUI(m); }
  let off = 4 + hlen;
  for (const ch of [...m.obj.children]) { m.obj.remove(ch); ch.geometry.dispose(); }
  for (const mesh of header.meshes) {
    const verts = new Float32Array(buf, off, mesh.nverts * 3); off += mesh.nverts * 12;
    const edges = new Uint32Array(buf, off, mesh.nedges); off += mesh.nedges * 4;
    if (!mesh.nverts) continue;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(verts.slice(), 3));
    g.setIndex(new THREE.BufferAttribute(edges.slice(), 1));
    let color = m.info.color;
    if (m.info.kind === "diff") color = mesh.sign > 0 ? "#00d000" : "#e00000";
    m.obj.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color })));
  }
  m.lastCenter = c; m.lastLevel = level;
  redraw();
}

// ------------------------------------------------------------- navigation
let anim = null;
function moveTo(target) {
  anim = { from: S.center.clone(), to: target.clone(), t0: performance.now(), dur: 350 };
}
function animateCenter() {
  if (!anim) return;
  const t = Math.min(1, (performance.now() - anim.t0) / anim.dur);
  const e = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
  S.center.lerpVectors(anim.from, anim.to, e);
  needsRender = true;
  if (t >= 1) { anim = null; scheduleContour(); pushView(); }
}
function setCenterNow(v) { S.center.copy(v); redraw(); scheduleContour(); pushView(); }

function currentResidue() {
  const mol = S.mols.get(S.clicked?.mol ?? S.activeMol);
  if (!mol) return null;
  if (S.clicked) return { mol, res: mol.residues.find((r) => r.atoms.includes(S.clicked.idx)) };
  let best = null, bd = Infinity;
  for (const r of mol.residues) {
    const i = r.ca, dx = mol.xyz[3*i] - S.center.x, dy = mol.xyz[3*i+1] - S.center.y, dz = mol.xyz[3*i+2] - S.center.z;
    const d = dx*dx + dy*dy + dz*dz;
    if (d < bd) { bd = d; best = r; }
  }
  return { mol, res: best };
}
function stepResidue(dir) {
  const cr = currentResidue();
  if (!cr || !cr.res) return;
  const list = cr.mol.residues;
  let k = list.indexOf(cr.res) + dir;
  if (k < 0 || k >= list.length) return;
  const r = list[k];
  S.clicked = { mol: cr.mol.data.id, idx: r.ca };
  const x = cr.mol.xyz;
  moveTo(new THREE.Vector3(x[3*r.ca], x[3*r.ca+1], x[3*r.ca+2]));
  status(atomInfo(cr.mol.data.id, r.ca));
}

// screen projection / picking
function projectAtom(mol, i, out) {
  out.set(mol.xyz[3*i], mol.xyz[3*i+1], mol.xyz[3*i+2]).applyMatrix4(world.matrixWorld).project(camera);
  return out;
}
function pick(clientX, clientY, maxPx = 12, onlyMoving = false) {
  updateCamera(); scene.updateMatrixWorld();
  const rect = renderer.domElement.getBoundingClientRect();
  const v = new THREE.Vector3();
  let best = null, bd = maxPx * maxPx;
  const consider = (molId, xyz, i) => {
    v.set(xyz[3*i], xyz[3*i+1], xyz[3*i+2]).applyMatrix4(world.matrixWorld).project(camera);
    if (v.z < -1 || v.z > 1) return;
    const sx = (v.x + 1) / 2 * rect.width, sy = (1 - v.y) / 2 * rect.height;
    const d = (sx - (clientX - rect.left)) ** 2 + (sy - (clientY - rect.top)) ** 2;
    if (d < bd || (best && Math.abs(d - bd) < 4 && v.z < best.z)) { bd = d; best = { mol: molId, idx: i, z: v.z }; }
  };
  if (S.rsr && S.rsr.xyz) {
    const full = S.rsr.full;
    for (const i of S.rsr.idx) consider(S.rsr.mol, full, i);
    if (best || onlyMoving) return best;
  }
  for (const [id, mol] of S.mols) {
    if (!mol.group.visible) continue;
    const el = mol.data.element;
    for (let i = 0; i < mol.data.atoms.length; i++) if (!(mol.hideH && el[i] === "H")) consider(id, mol.xyz, i);
  }
  return best;
}

// labels
function toggleLabel(molId, idx) {
  const k = `${molId}:${idx}`;
  if (S.labels.has(k)) { S.labels.get(k).remove(); S.labels.delete(k); }
  else {
    const div = document.createElement("div");
    div.className = "label"; div.textContent = atomLabel(molId, idx);
    $("#labels").appendChild(div); S.labels.set(k, div);
  }
  redraw();
}
function addLabel(molId, idx) {
  if (!S.labels.has(`${molId}:${idx}`) && S.mols.get(molId)) toggleLabel(molId, idx);
}
function annotate(m) {
  if (m.clear_labels) { for (const div of S.labels.values()) div.remove(); S.labels.clear(); }
  if (m.clear_distances) { for (const d of S.distances) { d.div.remove(); overlay.remove(d.line); } S.distances = []; }
  for (const l of m.labels || []) addLabel(l.mol, l.idx);
  for (const d of m.distances || []) if (S.mols.get(d.a.mol) && S.mols.get(d.b.mol)) addDistance(d.a, d.b, d.style, d.label);
  if (S.preview) previewDistances();
  redraw();
}
function updateLabels() {
  const rect = renderer.domElement.getBoundingClientRect();
  const v = new THREE.Vector3();
  scene.updateMatrixWorld();
  for (const [k, div] of S.labels) {
    const [mid, idx] = k.split(":").map(Number);
    const mol = S.mols.get(mid);
    if (!mol) continue;
    projectAtom(mol, idx, v);
    const vis = v.z > -1 && v.z < 1;
    div.style.display = vis ? "" : "none";
    div.style.left = `${(v.x + 1) / 2 * rect.width}px`;
    div.style.top = `${(1 - v.y) / 2 * rect.height}px`;
  }
  for (const d of S.distances) {
    const a = S.mols.get(d.a.mol), b = S.mols.get(d.b.mol);
    if (!a || !b) continue;
    const pa = projectAtom(a, d.a.idx, new THREE.Vector3()), pb = projectAtom(b, d.b.idx, new THREE.Vector3());
    d.div.style.left = `${((pa.x + pb.x) / 2 + 1) / 2 * rect.width}px`;
    d.div.style.top = `${(1 - (pa.y + pb.y) / 2) / 2 * rect.height + 11}px`;   // just below the line, clear of atom labels
  }
}
function clearLabels() {
  for (const div of S.labels.values()) div.remove();
  S.labels.clear();
  for (const d of S.distances) { d.div.remove(); overlay.remove(d.line); }
  S.distances = [];
  redraw();
}
function addDistance(a, b, style = "distance", showLabel = true) {
  const A = S.mols.get(a.mol).xyz, B = S.mols.get(b.mol).xyz;
  const pa = new THREE.Vector3(A[3*a.idx], A[3*a.idx+1], A[3*a.idx+2]);
  const pb = new THREE.Vector3(B[3*b.idx], B[3*b.idx+1], B[3*b.idx+2]);
  const g = new THREE.BufferGeometry().setFromPoints([pa, pb]);
  const hb = style === "hbond";
  const line = new THREE.LineSegments(g, new THREE.LineDashedMaterial({ color: hb ? 0x6fa8ff : 0x99ffff, dashSize: hb ? 0.12 : 0.2, gapSize: hb ? 0.12 : 0.15 }));
  line.computeLineDistances();
  overlay.add(line);
  const div = document.createElement("div");
  div.className = "label dist"; div.textContent = `${pa.distanceTo(pb).toFixed(2)} Å`;
  $("#labels").appendChild(div);
  if (!showLabel) div.style.visibility = "hidden";
  S.distances.push({ a, b, line, div, style, label: showLabel });
  status(`${atomLabel(a.mol, a.idx)} – ${atomLabel(b.mol, b.idx)}: ${pa.distanceTo(pb).toFixed(3)} Å`);
  redraw();
}

// -------------------------------------------------------------- refinement
function startRSR(m) {
  const mol = S.mols.get(m.mol);
  S.rsr = { mol: m.mol, idx: m.idx, set: new Set(m.idx), full: Float32Array.from(mol.xyz), xyz: true, obj: null };
  $("#refine-banner").classList.remove("hidden");
  $("#refine-text").textContent = `Refining ${m.idx.length} atoms${m.altloc ? " (alt " + m.altloc + ")" : ""} – drag atoms to pull`;
  drawRSR();
}
function rsrCoords(m) {
  if (!S.rsr || S.rsr.mol !== m.mol) return;
  const f = S.rsr.full;
  m.idx.forEach((i, k) => { f[3*i] = m.xyz[3*k]; f[3*i+1] = m.xyz[3*k+1]; f[3*i+2] = m.xyz[3*k+2]; });
  drawRSR();
}
function drawRSR() {
  const r = S.rsr;
  if (!r) return;
  if (r.obj) { overlay.remove(r.obj); r.obj.geometry.dispose(); }
  const mol = S.mols.get(r.mol);
  const b = bondGeometry(r.full, mol.data.bonds, mol.data.element, r.mol, r.set, INTERMEDIATE);
  r.obj = makeLines(b.pos, b.col, 3.5);
  overlay.add(r.obj);
  redraw();
}
function rsrStatus(m) {
  if (!S.rsr) return;
  $("#refine-text").textContent = `Refining ${S.rsr.idx.length} atoms · bonds ${m.bond_rmsd?.toFixed(3)} Å · ` +
    `angles ${m.angle_rmsd_deg?.toFixed(1)}°` +
    (m.mean_density_sigma != null ? ` · density ${m.mean_density_sigma.toFixed(2)}σ` : "") +
    (m.diff_cc != null ? ` · Δ CC ${m.diff_cc.toFixed(2)}` : "");
}
function endRSR() {
  if (S.rsr?.obj) { overlay.remove(S.rsr.obj); S.rsr.obj.geometry.dispose(); }
  S.rsr = null;
  $("#refine-banner").classList.add("hidden");
  redraw();
}

// ------------------------------------------------------------------- mouse
const el = renderer.domElement;
let drag = null;
el.addEventListener("contextmenu", (e) => e.preventDefault());
el.addEventListener("mousedown", (e) => {
  e.preventDefault();
  if (document.activeElement && document.activeElement !== document.body) document.activeElement.blur();
  drag = { x: e.clientX, y: e.clientY, button: e.button, ctrl: e.ctrlKey || e.metaKey, moved: false, pull: null };
  if (e.button === 0 && S.rsr && !drag.ctrl) {
    const p = pick(e.clientX, e.clientY, 12, true);
    if (p) {
      const f = S.rsr.full;
      const w = new THREE.Vector3(f[3*p.idx], f[3*p.idx+1], f[3*p.idx+2]).applyMatrix4(world.matrixWorld).project(camera);
      drag.pull = { idx: p.idx, ndcZ: w.z, last: 0 };
      status("pulling " + atomLabel(p.mol, p.idx));
    }
  }
});
window.addEventListener("mousemove", (e) => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  if (Math.abs(dx) + Math.abs(dy) > 2) drag.moved = true;
  if (!drag.moved) return;
  drag.x = e.clientX; drag.y = e.clientY;
  if (drag.pull) {
    const now = performance.now();
    if (now - drag.pull.last < 30) return;
    drag.pull.last = now;
    const rect = el.getBoundingClientRect();
    const ndc = new THREE.Vector3((e.clientX - rect.left) / rect.width * 2 - 1, -(e.clientY - rect.top) / rect.height * 2 + 1, drag.pull.ndcZ);
    ndc.unproject(camera);
    const inv = new THREE.Matrix4().copy(world.matrixWorld).invert();
    ndc.applyMatrix4(inv);
    send({ type: "pull", atom: drag.pull.idx, xyz: ndc.toArray() });
    return;
  }
  if (drag.button === 2) {
    S.zoom = Math.min(400, Math.max(4, S.zoom * Math.exp(-dy * 0.01)));
    redraw();
  } else if (drag.button === 1 || (drag.button === 0 && drag.ctrl)) {
    const h = 2 * S.zoom * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
    const s = h / el.clientHeight;
    const v = new THREE.Vector3(-dx * s, dy * s, 0).applyQuaternion(rot.quaternion.clone().invert());
    S.center.add(v);
    redraw(); scheduleContour();
  } else if (drag.button === 0) {
    const axis = new THREE.Vector3(dy, dx, 0);
    const ang = axis.length() * 0.008;
    if (ang > 0) {
      axis.normalize();
      rot.quaternion.premultiply(new THREE.Quaternion().setFromAxisAngle(axis, ang));
      redraw();
    }
  }
});
window.addEventListener("mouseup", (e) => {
  if (!drag) return;
  const d = drag;
  drag = null;
  if (d.pull) { send({ type: "pull", atom: d.pull.idx, xyz: null }); return; }
  if (d.moved) { pushView(); return; }
  if (e.target !== el) return;
  const p = pick(e.clientX, e.clientY);
  if (d.button === 1) { if (p) centerOnAtom(p); return; }
  if (d.button === 0 && p) onAtomClick(p, e);
});
el.addEventListener("dblclick", (e) => { const p = pick(e.clientX, e.clientY); if (p) centerOnAtom(p); });
el.addEventListener("wheel", (e) => {
  e.preventDefault();
  if (e.ctrlKey) { S.zoom = Math.min(400, Math.max(4, S.zoom * Math.exp(e.deltaY * 0.002))); redraw(); return; }
  const m = S.maps.get(S.scrollMap);
  if (!m) return;
  const step = (e.shiftKey ? 0.5 : 0.1) * (e.deltaY > 0 ? -1 : 1);
  changeLevel(m, step);
}, { passive: false });

function changeLevel(m, step) { setLevel(m, m.info.level_sigma + step); }

function centerOnAtom(p) {
  const x = p.mol === S.rsr?.mol && S.rsr.set.has(p.idx) ? S.rsr.full : S.mols.get(p.mol).xyz;
  moveTo(new THREE.Vector3(x[3*p.idx], x[3*p.idx+1], x[3*p.idx+2]));
  S.clicked = p;
  status(atomInfo(p.mol, p.idx));
}

function residueOf(p) {
  const a = S.mols.get(p.mol).data.atoms[p.idx];
  return { chain: a[0], resseq: a[1], resname: a[3], altloc: a[5], mol: p.mol };
}

// ------------------------------------------------------------------- tools
const TOOL_HELP = {
  refine_zone: "Refine zone: click the first atom, then the last atom",
  refine_triple: "Refine triple: click an atom",
  refine_sphere: "Sphere refine: click an atom",
  altconf: "Alt conf: click an atom of the residue to split",
  rotamers: "Rotamers: click a residue",
  mutate: "Mutate: click a residue",
  flip: "Pepflip: click an atom of residue i (flips peptide i/i+1)",
  add_residue: "Add residue: click an atom of the terminal residue",
  fix: "Fix atoms: click atoms to toggle (Shift-click: whole residue)",
  delete: "Delete: click a residue (Shift: atom only, Alt: side chain only)",
  distance: "Distance: click two atoms",
};
function setTool(t) {
  S.tool = S.tool === t ? null : t;
  S.toolClicks = [];
  document.querySelectorAll("#toolbar [data-tool]").forEach((b) => b.classList.toggle("active", b.dataset.tool === S.tool));
  const mb = $("#mode-banner");
  if (S.tool) { mb.textContent = TOOL_HELP[S.tool] + "  ·  Esc cancels"; mb.classList.remove("hidden"); }
  else mb.classList.add("hidden");
}
document.querySelectorAll("#toolbar [data-tool]").forEach((b) => b.addEventListener("click", () => setTool(b.dataset.tool)));

async function onAtomClick(p, e) {
  S.clicked = p;
  status(atomInfo(p.mol, p.idx));
  const r = residueOf(p);
  const tool = S.tool;
  if (!tool) {
    toggleLabel(p.mol, p.idx);
    const k = S.selection.findIndex((s) => s.mol === p.mol && s.idx === p.idx);
    if (k >= 0) S.selection.splice(k, 1); else S.selection.push(p);
    S.selection = S.selection.slice(-20);
    if (p.mol !== S.activeMol) { S.activeMol = p.mol; cmd("set_active_molecule", { mol: p.mol }).catch(() => {}); }
    pushView();
    if (S.preview) refreshPreviewSoon();
    return;
  }
  S.toolClicks.push(p);
  try {
    if (tool === "refine_zone") {
      if (S.toolClicks.length < 2) { status("first atom: " + atomLabel(p.mol, p.idx) + " – now click the last atom"); return; }
      const a = residueOf(S.toolClicks[0]), b = residueOf(S.toolClicks[1]);
      if (a.chain !== b.chain) throw new Error("zone must be within one chain");
      setTool(null);
      await cmd("refine", { chain: a.chain, start: Math.min(a.resseq, b.resseq), end: Math.max(a.resseq, b.resseq), altloc: a.altloc || b.altloc || null, mol: p.mol });
    } else if (tool === "refine_triple") {
      setTool(null); await refineTriple(r);
    } else if (tool === "refine_sphere") {
      setTool(null); await cmd("refine", { chain: r.chain, start: r.resseq, sphere_radius: 4.5, mol: p.mol });
    } else if (tool === "altconf") {
      setTool(null);
      const alt = await cmd("add_altconf", { chain: r.chain, resseq: r.resseq, mol: p.mol });
      await cmd("refine", { chain: r.chain, start: r.resseq, end: r.resseq, altloc: alt, mol: p.mol });
    } else if (tool === "rotamers") {
      setTool(null); await rotamerPopup(r);
    } else if (tool === "mutate") {
      setTool(null); mutatePopup(r);
    } else if (tool === "flip") {
      setTool(null); await cmd("flip_peptide", { chain: r.chain, resseq: r.resseq, mol: p.mol });
    } else if (tool === "add_residue") {
      setTool(null); await addResidueAt(p);
    } else if (tool === "fix") {
      const mol = S.mols.get(p.mol);
      const idx = e.shiftKey ? mol.residues.find((x) => x.atoms.includes(p.idx)).atoms : [p.idx];
      const fix = !mol.fixed.has(p.idx);
      send({ type: "fix", idx, fix });
    } else if (tool === "delete") {
      setTool(null);
      if (e.shiftKey) await cmd("delete", { chain: r.chain, start: r.resseq, atom: S.mols.get(p.mol).data.atoms[p.idx][4], mol: p.mol });
      else if (e.altKey) await cmd("delete", { chain: r.chain, start: r.resseq, side_chain_only: true, mol: p.mol });
      else await cmd("delete", { chain: r.chain, start: r.resseq, mol: p.mol });
    } else if (tool === "distance") {
      if (S.toolClicks.length === 2) {      // through the server, so views, tours and figures keep it
        const [a, b] = S.toolClicks;
        S.toolClicks = [];
        await cmd("show_distance", { a: `${a.mol}#${atomLabel(a.mol, a.idx)}`, b: `${b.mol}#${atomLabel(b.mol, b.idx)}` });
      }
    }
  } catch (err) { status("⚠ " + err.message); }
}

async function refineTriple(r) {
  await cmd("refine", { chain: r.chain, start: r.resseq - 1, end: r.resseq + 1, altloc: r.altloc || null, mol: r.mol });
}

async function addResidueAt(p) {
  const mol = S.mols.get(p.mol);
  const r = residueOf(p);
  const chainRes = mol.residues.filter((x) => x.chain === r.chain && mol.data.atoms[x.ca][4] === "CA");
  const first = chainRes[0], last = chainRes[chainRes.length - 1];
  const terminus = first && first.num === r.resseq ? "N" : "C";
  if (last && terminus === "C" && last.num !== r.resseq) status("note: not the last residue – adding after the chain's C terminus");
  const res = await cmd("add_residue", { chain: r.chain, terminus, mol: p.mol });
  await cmd("refine", { chain: r.chain, start: Math.min(res.resseq, r.resseq), end: Math.max(res.resseq, r.resseq), mol: p.mol });
}

function popup(html) {
  const pp = $("#popup");
  pp.innerHTML = html;
  pp.classList.remove("hidden");
  return pp;
}
const closePopup = () => $("#popup").classList.add("hidden");

async function rotamerPopup(r) {
  const res = await cmd("rotamers", { chain: r.chain, resseq: r.resseq, altloc: r.altloc || "", mol: r.mol });
  if (!res || !res.rotamers) { status(`no rotamers for ${r.resname}`); return; }
  let applied = 0;
  const pp = popup(`<h4>${r.chain}/${r.resseq} ${r.resname} rotamers</h4>` +
    `<div class="hint">current χ: ${res.current_chis.map((c) => c.toFixed(0)).join(", ")}</div>` +
    res.rotamers.map((x, i) => `<div class="opt" data-i="${i}"><span>${x.name}</span><span>${x.density == null ? "" : x.density.toFixed(2) + "σ"}</span></div>`).join("") +
    `<div class="row" style="margin-top:6px"><button id="rot-ok">Accept</button><button id="rot-refine">Accept + refine</button><button id="rot-cancel">Cancel</button></div>` +
    `<div class="hint">, and . cycle</div>`);
  let cur = -1;
  const choose = async (i) => {
    if (i < 0 || i >= res.rotamers.length) return;
    cur = i;
    pp.querySelectorAll(".opt").forEach((o) => o.classList.toggle("sel", +o.dataset.i === i));
    if (applied) { await cmd("undo", { mol: r.mol }); applied--; }
    await cmd("set_rotamer", { chain: r.chain, resseq: r.resseq, name: res.rotamers[i].name, altloc: r.altloc || "", mol: r.mol });
    applied++;
  };
  pp.querySelectorAll(".opt").forEach((o) => o.addEventListener("click", () => choose(+o.dataset.i)));
  S.popupKeys = (k) => { if (k === ".") choose(cur + 1); else if (k === ",") choose(cur - 1); };
  $("#rot-ok").onclick = () => { closePopup(); S.popupKeys = null; };
  $("#rot-refine").onclick = async () => { closePopup(); S.popupKeys = null; await refineTriple(r); };
  $("#rot-cancel").onclick = async () => { closePopup(); S.popupKeys = null; if (applied) await cmd("undo", { mol: r.mol }); };
  choose(0);
}

const AA = ["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"];
function mutatePopup(r) {
  const pp = popup(`<h4>Mutate ${r.chain}/${r.resseq} ${r.resname}</h4>` +
    AA.map((a) => `<div class="opt" data-a="${a}"><span>${a}</span></div>`).join("") +
    `<div class="row"><button id="mut-cancel">Cancel</button></div>`);
  pp.querySelectorAll(".opt").forEach((o) => o.addEventListener("click", async () => {
    closePopup();
    await cmd("mutate", { chain: r.chain, resseq: r.resseq, resname: o.dataset.a, mol: r.mol });
  }));
  $("#mut-cancel").onclick = closePopup;
}

function promptPopup(title, fields, onOk) {
  const pp = popup(`<h4>${title}</h4>` + fields.map((f) =>
    `<div class="row"><label style="width:70px">${f.label}</label><input data-k="${f.key}" value="${f.value ?? ""}" style="flex:1"></div>`).join("") +
    `<div class="row"><button id="pp-ok">OK</button><button id="pp-cancel">Cancel</button></div>`);
  const first = pp.querySelector("input"); first && first.focus();
  const ok = async () => {
    const v = {};
    pp.querySelectorAll("input").forEach((i) => (v[i.dataset.k] = i.value.trim()));
    closePopup();
    try { await onOk(v); } catch (e) { status("⚠ " + e.message); }
  };
  $("#pp-ok").onclick = ok;
  $("#pp-cancel").onclick = closePopup;
  pp.querySelectorAll("input").forEach((i) => i.addEventListener("keydown", (e) => { if (e.key === "Enter") ok(); if (e.key === "Escape") closePopup(); }));
}

$("#btn-water").onclick = () => cmd("add_water");
$("#btn-ion").onclick = () => promptPopup("Add ion at centre", [{ key: "element", label: "element", value: "MG" }], (v) => cmd("add_ion", { element: v.element }));
$("#btn-ligand").onclick = () => promptPopup("Add ligand at centre", [{ key: "cif", label: "CIF path" }, { key: "comp_id", label: "comp id" }],
  (v) => cmd("add_ligand", { cif: v.cif, comp_id: v.comp_id || null }));
$("#btn-renumber").onclick = () => {
  const cr = currentResidue();
  promptPopup("Renumber residues", [{ key: "chain", label: "chain", value: cr?.res?.chain ?? "A" },
    { key: "start", label: "from", value: cr?.res?.num ?? "" }, { key: "end", label: "to", value: cr?.res?.num ?? "" },
    { key: "offset", label: "offset", value: "1" }],
  (v) => cmd("renumber", { chain: v.chain, start: +v.start, end: +v.end, offset: +v.offset }));
};
$("#btn-undo").onclick = () => cmd("undo");
$("#btn-redo").onclick = () => cmd("redo");
$("#btn-save").onclick = () => {
  const mol = S.mols.get(S.activeMol);
  const def = mol?.data.path ? mol.data.path.replace(/(\.[^.]+)$/, "_coot$1") : "model.pdb";
  promptPopup("Save model", [{ key: "path", label: "path", value: def }], async (v) => status("saved " + await cmd("save_model", { path: v.path })));
};
function openFigureDialog(tour) {
  const dlg = $("#figure-dlg");
  dlg.dataset.mode = tour ? "tour" : "single";
  $("#figure-head").textContent = tour ? "Figures of every tour stop" : "Publication figure";
  dlg.querySelectorAll(".single-only").forEach((e) => e.classList.toggle("hidden", !!tour));
  dlg.classList.remove("hidden");
}
const selSpecs = () => S.selection.map((p) => `${p.mol}#${atomLabel(p.mol, p.idx)}`);
$("#btn-zone").onclick = async () => {
  if (!S.selection.length) { status("Click atoms first (they get labels); the zone is drawn around them."); return; }
  const r = await cmd("show_zone", { atoms: selSpecs(), radius: +$("#zone-radius").value || 5 });
  status(`Showing ${Object.values(r.residues).reduce((n, x) => n + x.length, 0)} residues within ${r.radius} Å of the clicked atoms.`);
};
$("#btn-zone-clear").onclick = () => cmd("show_zone", { clear: true });
$("#btn-hbonds").onclick = async () => {
  if (!S.selection.length) { status("Click an atom of each residue whose H-bonds you want."); return; }
  const res = [...new Set(S.selection.map((p) => { const a = S.mols.get(p.mol).data.atoms[p.idx]; return `${p.mol}#${a[0]}/${a[1]}${a[2]}`; }))];
  const r = await cmd("show_hbonds", { atoms: res });
  status(r.length ? `${r.length} H-bonds: ` + r.slice(0, 6).map((h) => `${h.a} – ${h.b} ${h.distance} Å`).join("; ") : "No N/O contacts within 3.3 Å.");
};
$("#btn-figure").onclick = () => openFigureDialog(false);
$("#tour-figures").onclick = () => openFigureDialog(true);
$("#figure-cancel").onclick = () => $("#figure-dlg").classList.add("hidden");
$("#figure-form").onsubmit = async (e) => {
  e.preventDefault();
  const f = new FormData(e.target), args = {};
  for (const [k, v] of f.entries()) if (v !== "") args[k] = v;
  args.dpi = +args.dpi; args.carve = +args.carve; args.pymol = f.has("pymol"); args.look = f.has("look");
  args.look_radius = +args.look_radius || 5;
  if (args.levels) {
    const lv = args.levels.split(/[ ,;]+/).map(Number).filter((x) => x > 0);
    if (lv.length) args.levels = lv; else delete args.levels;
  }
  const tour = $("#figure-dlg").dataset.mode === "tour";
  $("#figure-dlg").classList.add("hidden");
  status(tour ? "Making one figure per tour stop…" : "Making the figure…");
  try {
    if (tour) {
      for (const k of ["name", "text"]) delete args[k];
      const r = await cmd("figure_tour", { options: args });
      status(`Figures: ${r.figures.length} in ${r.folder}`);
      addLog(`figures written to ${r.folder} (legends in figures.md)`);
      for (const fg of r.figures) for (const w of fg.warnings) addLog(`⚠ ${fg.stop}: ${w}`);
    } else {
      const r = await cmd("figure", args);
      status(`Figure: ${r.files.png}${r.warnings.length ? "  ⚠ " + r.warnings.join("; ") : ""}`);
      addLog(`figure written: ${Object.values(r.files).join(", ")}`);
      for (const w of r.warnings) addLog("⚠ " + w);
    }
  } catch (err) { /* cmd shows the error */ }
};
$("#btn-open").onclick = () => promptPopup("Open file or TorchRef directory", [{ key: "path", label: "path" }], async (v) => {
  const p = v.path;
  if (/\.mtz$/i.test(p)) await cmd("load_mtz", { path: p });
  else if (/\.(map|ccp4|mrc)$/i.test(p)) await cmd("load_map", { path: p });
  else if (/\.(pdb|ent|cif|mmcif)$/i.test(p)) await cmd("load_model", { path: p });
  else await cmd("load_torchref", { path: p });
});
$("#btn-accept").onclick = () => cmd("accept_refinement");
$("#btn-reject").onclick = () => cmd("reject_refinement");
$("#btn-help").onclick = () => $("#help").classList.remove("hidden");
$("#help").onclick = () => $("#help").classList.add("hidden");
$("#btn-clear-fixed").onclick = () => cmd("unfix_atoms");
$("#rsr-weight").onchange = (e) => cmd("set_refinement_options", { weight: +e.target.value });
$("#rsr-diff-weight").onchange = (e) => cmd("set_refinement_options", { diff_weight: +e.target.value });
$("#rsr-target").onchange = (e) => cmd("set_refinement_options", { target: e.target.value });
$("#rsr-diff-map").onchange = (e) => cmd("set_refinement_options", { diff_map: +e.target.value });
$("#rsr-dark").onchange = (e) => cmd("set_refinement_options", { dark_model: +e.target.value });
$("#rsr-nonbonded").onchange = (e) => cmd("set_refinement_options", { nonbonded: e.target.checked });

function setRsrSettings(st) {
  S.rsrSettings = st;
  $("#rsr-target").value = st.target;
  $("#rsr-weight").value = st.weight;
  $("#rsr-diff-weight").value = st.diff_weight;
  $("#rsr-nonbonded").checked = st.nonbonded;
  renderRsrSelects();
  renderMapList();
}
function renderRsrSelects() {
  const st = S.rsrSettings || {};
  const dm = $("#rsr-diff-map");
  dm.innerHTML = `<option value="0">—</option>` + [...S.maps.values()].filter((m) => m.info.kind === "diff")
    .map((m) => `<option value="${m.info.id}">[${m.info.id}] ${m.info.name}</option>`).join("");
  dm.value = st.diff_map ?? 0;
  const dk = $("#rsr-dark");
  dk.innerHTML = `<option value="0">other altlocs / start coords</option>` + [...S.mols.values()]
    .map((m) => `<option value="${m.data.id}">${m.data.id}: ${m.data.name}</option>`).join("");
  dk.value = st.dark_model ?? 0;
}

function gotoText(t) {
  const m = t.trim().match(/^([A-Za-z0-9]+)?[\s/:]*(-?\d+)([A-Za-z]?)$/);
  if (!m) { status("format: A 45"); return; }
  const chain = m[1] ?? (currentResidue()?.res?.chain ?? "A");
  cmd("goto", { chain, resseq: +m[2], icode: m[3] || "" }).then(() => {
    const mol = S.mols.get(S.activeMol);
    const r = mol?.residues.find((x) => x.chain === chain && x.num === +m[2]);
    if (r) { S.clicked = { mol: S.activeMol, idx: r.ca }; status(atomInfo(S.activeMol, r.ca)); }
  }).catch(() => {});
}
$("#btn-goto").onclick = () => gotoText($("#goto").value);

// ligand navigation: list fetched per molecule, refreshed after model changes
let ligCache = null, ligIndex = -1;
async function stepLigand(dir) {
  if (!ligCache || ligCache.mol !== S.activeMol) {
    ligCache = { mol: S.activeMol, list: await cmd("ligands", { mol: S.activeMol }) };
  }
  const list = $("#lig-all").checked ? ligCache.list : ligCache.list.filter((l) => l.relevant);
  if (!list.length) { status(ligCache.list.length ? "only additives here – tick 'incl. additives'" : "no ligands"); return; }
  ligIndex = (ligIndex + dir + list.length) % list.length;
  const l = list[ligIndex];
  moveTo(new THREE.Vector3(...l.center));
  const mol = S.mols.get(S.activeMol);
  const r = mol?.residues.find((x) => x.chain === l.chain && x.num === l.resseq && x.icode === l.icode);
  if (r) S.clicked = { mol: S.activeMol, idx: r.atoms[0] };
  status(`ligand ${ligIndex + 1}/${list.length}: ${l.chain}/${l.resseq}${l.icode} ${l.resname}` +
    (l.relevant ? "" : `  (${l.reason})`) + (l.reason.startsWith("metal") ? `  – ${l.reason}` : ""));
}
$("#btn-lig-next").onclick = () => stepLigand(1);
$("#btn-lig-prev").onclick = () => stepLigand(-1);
$("#lig-all").onchange = () => { ligIndex = -1; };
$("#goto").addEventListener("keydown", (e) => { if (e.key === "Enter") gotoText(e.target.value); });

// --------------------------------------------------------------- keyboard
window.addEventListener("keydown", async (e) => {
  const tag = document.activeElement?.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
  const k = e.key;
  if (S.popupKeys && (k === "," || k === ".")) { S.popupKeys(k); return; }
  const cr = () => (S.clicked ? residueOf(S.clicked) : null);
  try {
    if (k === " ") { e.preventDefault(); stepResidue(e.shiftKey ? -1 : 1); }
    else if (k === "Enter" && S.rsr) await cmd("accept_refinement");
    else if (k === "Escape") {
      if (S.rsr) await cmd("reject_refinement");
      else if (S.tool) setTool(null);
      else { closePopup(); $("#help").classList.add("hidden"); }
    }
    else if ((k === "g" || k === "G") && !e.metaKey) { e.preventDefault(); $("#goto").focus(); $("#goto").select(); }
    else if (k === "+" || k === "=") { const m = S.maps.get(S.scrollMap); m && changeLevel(m, 0.1); }
    else if (k === "-" || k === "_") { const m = S.maps.get(S.scrollMap); m && changeLevel(m, -0.1); }
    else if (k === "n" || k === "N") { S.zoom = Math.max(4, S.zoom * 0.9); redraw(); }
    else if (k === "m" || k === "M") { S.zoom = Math.min(400, S.zoom * 1.1); redraw(); }
    else if (k === "d" || k === "D") { S.slab = Math.max(1, S.slab * 0.85); redraw(); pushView(); }
    else if (k === "f" || k === "F") { S.slab = Math.min(200, S.slab * 1.15); redraw(); pushView(); }
    else if (k === "c" || k === "C") { S.cross = !S.cross; redraw(); }
    else if (k === "l" || k === "L") clearLabels();
    else if (k === "?") $("#help").classList.toggle("hidden");
    else if ((k === "z" || k === "Z") && (e.metaKey || e.ctrlKey)) { e.preventDefault(); await cmd(e.shiftKey ? "redo" : "undo"); }
    else if (k === "u" || k === "U") await cmd("undo");
    else if (k === "w" || k === "W") await cmd("add_water");
    else if (k === "r" && cr()) await refineTriple(cr());
    else if (k === "R" && cr()) { const r = cr(); await cmd("refine", { chain: r.chain, start: r.resseq, sphere_radius: 4.5, mol: r.mol }); }
    else if ((k === "y" || k === "Y") && S.clicked) await addResidueAt(S.clicked);
    else if ((k === "q" || k === "Q") && cr()) { const r = cr(); await cmd("flip_peptide", { chain: r.chain, resseq: r.resseq, mol: r.mol }); }
    else if ((k === "a" || k === "A") && cr()) {
      const r = cr();
      const res = await cmd("rotamers", { chain: r.chain, resseq: r.resseq, altloc: r.altloc || "", mol: r.mol });
      if (res?.rotamers?.length) {
        await cmd("set_rotamer", { chain: r.chain, resseq: r.resseq, name: res.rotamers[0].name, altloc: r.altloc || "", mol: r.mol });
        await refineTriple(r);
      }
    }
  } catch (err) { /* status already shows the error */ }
});

// ------------------------------------------------------------------ panels
function renderMolList() {
  renderUndo();
  const box = $("#mol-list");
  box.innerHTML = "";
  for (const [id, mol] of S.mols) {
    const d = mol.data;
    const div = document.createElement("div");
    div.className = "item" + (id === S.activeMol ? " active-mol" : "");
    const sw = `#${carbonColor(id).toString(16).padStart(6, "0")}`;
    div.innerHTML = `<input type="checkbox" ${d.visible !== false && mol.group.visible ? "checked" : ""}>` +
      `<span class="swatch" style="background:${sw}"></span><span class="name" title="${d.path ?? ""}">${id}: ${d.name}</span>` +
      `<span class="abs">${d.atoms.length} at</span>` +
      (d.element.includes("H") ? `<label title="show hydrogens"><input type="checkbox" ${mol.hideH ? "" : "checked"}>H</label>` : "") +
      `<button title="Style: sticks → cartoon (main chain as a tube, side chains as sticks) → lines">${mol.style || "sticks"}</button>` +
      `<button title="Make active (edits apply here)">edit</button><button title="Close">×</button>`;
    const [cb, hcb] = div.querySelectorAll("input");
    const swatch = div.querySelector(".swatch");
    swatch.title = "Change the carbon colour";
    swatch.style.cursor = "pointer";
    swatch.onclick = () => {
      const pick = document.createElement("input");
      pick.type = "color"; pick.value = sw;
      Object.assign(pick.style, { position: "fixed", left: "50%", top: "40%", opacity: "0", width: "1px", height: "1px" });
      document.body.appendChild(pick);
      pick.onchange = () => { cmd("set_molecule", { mol: id, color: pick.value }); pick.remove(); };
      pick.onblur = () => setTimeout(() => pick.remove(), 500);
      pick.click();
    };
    const styleBtn = div.querySelector("button[title^='Style:']");
    styleBtn.onclick = () => cmd("set_molecule", { mol: id, style: { sticks: "cartoon", cartoon: "lines", lines: "sticks" }[mol.style] || "sticks" });
    if (hcb) hcb.onchange = () => cmd("set_molecule", { mol: id, hydrogens: hcb.checked });
    const act = div.querySelector("button[title^='Make active']"), close = div.querySelector("button[title='Close']");
    cb.onchange = () => { mol.group.visible = cb.checked; redraw(); cmd("set_molecule", { mol: id, visible: cb.checked }); };
    act.onclick = () => cmd("set_active_molecule", { mol: id });
    close.onclick = () => cmd("close_molecule", { mol: id });
    box.appendChild(div);
  }
}

function renderMapList() {
  const box = $("#map-list");
  box.innerHTML = "";
  const sel = $("#peak-map");
  const prev = sel.value;
  sel.innerHTML = "";
  for (const [id, m] of S.maps) {
    const i = m.info;
    const wrap = document.createElement("div");
    wrap.className = "mapitem";
    const div = document.createElement("div");
    div.className = "item";
    const color = i.kind === "diff" ? "linear-gradient(90deg,#00d000 50%,#e00000 50%)" : i.color;
    div.innerHTML = `<input type="radio" name="scroll" title="scroll wheel changes this map" ${S.scrollMap === id ? "checked" : ""}>` +
      `<input type="checkbox" ${i.visible ? "checked" : ""}>` +
      `<span class="swatch" style="background:${color}"></span>` +
      `<span class="name"></span>` +
      `<span title="refinement map (target: map)" style="cursor:pointer;color:${S.refinementMap === id ? "#e6c229" : "#444"}">R</span>` +
      (i.kind === "diff"
        ? `<span title="difference map (target: difference)" style="cursor:pointer;color:${S.rsrSettings?.diff_map === id ? "#4fd0ff" : "#444"}">Δ</span>`
        : `<span style="width:0.7em"></span>`) +
      `<button title="Close">×</button>`;
    const nm = div.querySelector(".name");
    nm.textContent = `[${id}] ${i.name}`;
    nm.title = `${i.source}: ${i.f} / ${i.phi}`;
    div.querySelector("button[title='Close']").onclick = () => cmd("close_map", { map_id: id });
    const [radio, cb] = div.querySelectorAll("input");
    const rbtn = div.querySelector("span[title^='refinement map']");
    const dbtn = div.querySelector("span[title^='difference map']");
    if (dbtn) dbtn.onclick = () => cmd("set_refinement_options", { diff_map: id });
    radio.onchange = () => { S.scrollMap = id; pushView(); };
    cb.onchange = () => { i.visible = cb.checked; applyMapVisibility(); m.lastCenter = null; scheduleContour(true); pushView(); redraw(); };
    rbtn.onclick = () => cmd("set_map", { map_id: id, refinement_map: true });
    const lvl = document.createElement("div");
    lvl.className = "lvl";
    lvl.innerHTML = `<button type="button" aria-label="Lower contour level">&minus;</button>` +
      `<input type="range" min="0" max="1000" step="1">` +
      `<button type="button" aria-label="Raise contour level">+</button>` +
      `<input type="number" step="0.1" title="contour level (σ)">`;
    const [down, up] = lvl.querySelectorAll("button");
    const [range, num] = lvl.querySelectorAll("input");
    range.setAttribute("aria-label", `Contour level of ${i.name}`);
    down.onclick = () => changeLevel(m, -levelStep(m));
    up.onclick = () => changeLevel(m, levelStep(m));
    range.addEventListener("input", () => setLevel(m, posToLevel(m, +range.value), range));
    range.addEventListener("change", () => scheduleContour(true));
    num.onchange = () => setLevel(m, +num.value, num);
    const scale = document.createElement("div");
    scale.className = "scale";
    m.ui = { range, num, scale };
    wrap.append(div, lvl, scale);
    box.appendChild(wrap);
    updateLevelUI(m);
    const o = document.createElement("option");
    o.value = id; o.textContent = `[${id}] ${i.name}`;
    sel.appendChild(o);
  }
  if (prev) sel.value = prev;
  else { const d = [...S.maps.values()].find((m) => m.info.kind === "diff" && m.info.visible); if (d) sel.value = d.info.id; }
  renderSliceMapUI();
}

// Levels as in shared pages: sliders on a log scale from 1σ (difference maps) or 0.3σ (density) to the map's largest
// value, so the usual 1-6σ get most of the travel and the strongest peaks are still in reach.
const levelStep = (m) => (m.info.kind === "diff" ? 0.5 : 0.25);
const fmt3 = (v) => { const a = Math.abs(v); return a && (a < 1e-3 || a >= 1e5) ? v.toExponential(2) : String(+v.toPrecision(3)); };
function levelRange(m) {
  const lo = m.info.kind === "diff" ? 1 : 0.3;
  return [lo, Math.max(lo * 4, m.info.max_sigma || lo * 20)];
}
function levelToPos(m, v) {
  const [lo, hi] = levelRange(m);
  return Math.round(1000 * Math.log(Math.min(hi, Math.max(lo, v)) / lo) / Math.log(hi / lo));
}
function posToLevel(m, p) {
  const [lo, hi] = levelRange(m);
  const v = lo * Math.pow(hi / lo, p / 1000);
  return v < 10 ? Math.round(v * 10) / 10 : Math.round(v);
}
const fmtLevel = (m) => `${m.info.kind === "diff" ? "±" : ""}${m.info.level_sigma.toFixed(m.info.level_sigma < 10 ? 2 : 1)}σ`;
function updateLevelUI(m, source = null) {
  const u = m.ui, i = m.info;
  if (!u) return;
  if (u.range !== source) u.range.value = String(levelToPos(m, i.level_sigma));
  if (u.num !== source) u.num.value = i.level_sigma.toFixed(2);
  const diff = i.kind === "diff";
  u.scale.textContent = `map rms ${fmt3(i.rms)} · level ${diff ? "±" : ""}${fmt3(i.level_sigma * i.rms + (diff ? 0 : i.mean || 0))} (map units)`;
}
function setLevel(m, v, source = null) {
  if (!(v > 0)) return;
  m.info.level_sigma = Math.max(0.1, +(+v).toFixed(2));
  m.lastLevel = null;
  updateLevelUI(m, source);
  status(`${m.info.name}: ${fmtLevel(m)} (${fmt3(m.info.level_sigma * m.info.rms)} map units)`);
  scheduleContour(); pushView(); redraw();
}

// ------------------------------------------------------------------- tours
function renderTour(t) {
  const box = $("#tour");
  if (!t.stops || !t.stops.length) { box.classList.add("hidden"); return; }
  box.classList.remove("hidden");
  const i = Math.max(0, t.index);
  $("#tour-title").textContent = t.title || "Tour";
  $("#tour-pos").textContent = `${t.index + 1}/${t.stops.length}`;
  $("#tour-name").textContent = t.stops[i]?.name ?? "";
  $("#tour-text").innerHTML = marked.parse(t.stops[i]?.text ?? "");
  $("#tour-prev").disabled = t.index <= 0;
  $("#tour-next").disabled = t.index >= t.stops.length - 1;
}
$("#tour-prev").onclick = () => cmd("tour_goto", { step: -1 });
$("#tour-next").onclick = () => cmd("tour_goto", { step: 1 });

function addLog(t) {
  const lg = $("#log");
  lg.textContent += t + "\n";
  lg.scrollTop = lg.scrollHeight;
  status(t.replace(/^\S+ /, ""));
}
function status(t) { $("#status").textContent = t; }

document.querySelectorAll("#tabs button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll("#tabs button").forEach((x) => x.classList.toggle("active", x === b));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("hidden", t.id !== "tab-" + b.dataset.tab));
}));

// peaks
$("#btn-peaks").onclick = async () => {
  const id = +$("#peak-map").value;
  const list = await cmd("peaks", { map_id: id, sigma: +$("#peak-sigma").value, max_peaks: 60 });
  const box = $("#peak-list");
  box.innerHTML = list.length ? "" : "<div class='hint'>no peaks</div>";
  for (const p of list) {
    const d = document.createElement("div");
    d.className = "peak";
    d.innerHTML = `<span class="${p.height_sigma > 0 ? "pos" : "neg"}">${p.height_sigma > 0 ? "+" : ""}${p.height_sigma.toFixed(1)}σ</span>` +
      `<span>${p.nearest}</span><span>${p.distance.toFixed(1)} Å</span>`;
    d.onclick = () => moveTo(new THREE.Vector3(...p.xyz));
    box.appendChild(d);
  }
};
$("#btn-find-waters").onclick = async () => {
  const sites = await cmd("find_waters", { sigma: +$("#peak-sigma").value });
  const box = $("#peak-list");
  box.innerHTML = `<div class="row"><b>${sites.length} water sites</b><button id="add-all-w">Add all</button></div>`;
  for (const s of sites) {
    const d = document.createElement("div");
    d.className = "peak";
    d.innerHTML = `<span class="pos">+${s.height_sigma.toFixed(1)}σ</span><span>${s.dist_polar.toFixed(2)} Å to N/O</span><button>add</button>`;
    d.querySelector("button").onclick = (e) => { e.stopPropagation(); cmd("add_water", { x: s.xyz[0], y: s.xyz[1], z: s.xyz[2] }); d.remove(); };
    d.onclick = () => moveTo(new THREE.Vector3(...s.xyz));
    box.appendChild(d);
  }
  $("#add-all-w").onclick = () => cmd("find_waters", { sigma: +$("#peak-sigma").value, add: true });
};

// --------------------------------------------------------------------- chat
let streamDiv = null;
function chatAdd(m) {
  const log = $("#chat-log");
  if (m.role === "assistant") {
    if (m.streamed && streamDiv) { streamDiv.innerHTML = marked.parse(m.text); streamDiv = null; return; }
    streamDiv = null;
  }
  const d = document.createElement("div");
  d.className = "msg " + m.role;
  if (m.role === "assistant") d.innerHTML = marked.parse(m.text);
  else if (m.role === "tool") d.textContent = `⚙ ${m.text} ${m.input && m.input !== "{}" ? m.input : ""}`;
  else d.textContent = m.text;
  log.appendChild(d);
  log.scrollTop = log.scrollHeight;
}
function chatStreamStart() { streamDiv = null; }
function chatStreamDelta(t) {
  const log = $("#chat-log");
  if (!streamDiv) {
    streamDiv = document.createElement("div");
    streamDiv.className = "msg assistant";
    streamDiv._raw = "";
    log.appendChild(streamDiv);
  }
  streamDiv._raw += t;
  streamDiv.innerHTML = marked.parse(streamDiv._raw);
  log.scrollTop = log.scrollHeight;
}
function chatBusy(b) {
  $("#chat-send").classList.toggle("hidden", b);
  $("#chat-stop").classList.toggle("hidden", !b);
  if (b) $("#chat-status").textContent = "Claude is working…";
}
function chatSend() {
  const t = $("#chat-input").value.trim();
  if (!t) return;
  send({ type: "chat", text: t });
  $("#chat-input").value = "";
}
$("#chat-send").onclick = chatSend;
$("#chat-stop").onclick = () => send({ type: "chat_stop" });
$("#chat-new").onclick = () => send({ type: "chat_new" });
$("#chat-bash").onchange = (e) => send({ type: "chat_settings", settings: { allow_bash: e.target.checked } });
$("#chat-input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); chatSend(); } });

// debugging / automation hook
window.moorlight = {
  S, cmd, pick,
  screenPos(molId, idx, useRSR = true) {
    updateCamera(); scene.updateMatrixWorld();
    const x = useRSR && S.rsr && S.rsr.mol === molId ? S.rsr.full : S.mols.get(molId).xyz;
    const v = new THREE.Vector3(x[3*idx], x[3*idx+1], x[3*idx+2]).applyMatrix4(world.matrixWorld).project(camera);
    const r = el.getBoundingClientRect();
    return [r.left + (v.x + 1) / 2 * r.width, r.top + (1 - v.y) / 2 * r.height];
  },
};

// ------------------------------------------------------------------- figures
// Publication figures of the current view, rendered off screen at journal size: orthographic, smoother atoms with
// a little shine, maps as nested transparent surfaces (weaker levels more transparent), distances as dashed sticks,
// labels returned as text positions (the server writes them into an SVG) and drawn into one PNG.
const FIG = { sphere: new THREE.SphereGeometry(1, 32, 24), cyl: new THREE.CylinderGeometry(1, 1, 1, 28, 1, true),
  mat: new THREE.MeshPhongMaterial({ color: 0xffffff, shininess: 45, specular: 0x2a2a2a }) };

async function fetchFigureSurfaces(id, carved) {
  const buf = await (await fetch(`/api/figure/surfaces?map=${id}&carved=${carved ? 1 : 0}`)).arrayBuffer();
  const hlen = new DataView(buf).getUint32(0, true);
  const h = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hlen)));
  let off = 4 + hlen;
  return h.surfaces.map((sf) => {
    const v = new Float32Array(buf, off, sf.nverts * 3); off += sf.nverts * 12;
    const n = new Float32Array(buf, off, sf.nverts * 3); off += sf.nverts * 12;
    const f = new Uint32Array(buf, off, sf.nfaces * 3); off += sf.nfaces * 12;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(v.slice(), 3));
    g.setAttribute("normal", new THREE.BufferAttribute(n.slice(), 3));
    g.setIndex(new THREE.BufferAttribute(f.slice(), 1));
    return { sign: sf.sign, level: sf.level, geom: g };
  });
}
// surfaces of all figure maps; inner (strong) levels first and solid enough to hide what is behind, outer ones fainter
async function buildFigureMaps(o, carved) {
  const grp = new THREE.Group();
  for (const fm of o.maps) {
    const surfs = await fetchFigureSurfaces(fm.id, carved);
    const order = [...fm.levels].sort((a, b) => a - b);
    for (const sf of surfs) {
      const k = order.indexOf(sf.level), nk = order.length;
      const opacity = fm.opacity[Math.min(fm.opacity.length - 1, Math.round(k * (fm.opacity.length - 1) / Math.max(1, nk - 1)))];
      const color = fm.kind === "diff" ? (sf.sign > 0 ? fm.colors[0] : fm.colors[1]) : fm.colors[0];
      const mat = new THREE.MeshPhongMaterial({ color, transparent: true, opacity, side: THREE.DoubleSide, shininess: 30,
        specular: 0x202020, depthWrite: k === nk - 1, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 });
      const mesh = new THREE.Mesh(sf.geom, mat);
      mesh.renderOrder = 10 + (nk - 1 - k);       // innermost first
      grp.add(mesh);
    }
  }
  return grp;
}
function disposeGroup(g) {
  g.traverse((o) => { if (o.geometry) o.geometry.dispose(); if (o.material) o.material.dispose(); });
  g.parent?.remove(g);
}
// dashed sticks for distances and H-bonds (screen lines are one pixel wide, too thin at print resolution)
function figureDistances(bg) {
  const grp = new THREE.Group(), dark = bg !== "black";
  const pts = { distance: [], hbond: [] };
  for (const d of S.distances) {
    const A = S.mols.get(d.a.mol), B = S.mols.get(d.b.mol);
    if (A && B) pts[d.style === "hbond" ? "hbond" : "distance"].push(P3(A, d.a.idx), P3(B, d.b.idx));
  }
  if (pts.distance.length) grp.add(dashedSticks(pts.distance, dark ? 0x333333 : 0xbfefff, 0.045, 0.18, 0.14));
  if (pts.hbond.length) grp.add(dashedSticks(pts.hbond, dark ? 0x3f6fbf : 0x8fbfff, 0.035, 0.12, 0.12));
  overlay.add(grp);
  return grp;
}
const AA3 = new Set(["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO",
  "SER", "THR", "TRP", "TYR", "VAL", "KCX", "MSE", "SEP", "TPO", "PTR", "CSO", "LLP", "MLY", "HYP"]);
const title = (t) => t.charAt(0) + t.slice(1).toLowerCase();
// label text as in papers: Glu204, KCX201 (modified residues keep their code), Zn, W721, RBD; the atom name only when
// several labels sit in one residue
function paperLabel(mid, i, many) {
  const a = S.mols.get(mid).data.atoms[i], el = S.mols.get(mid).data.element[i];
  let t;
  if (METALS.has((el || "").toUpperCase()) && a[3] === a[4]) t = title(el.toUpperCase());
  else if (["HOH", "WAT", "DOD"].includes(a[3])) t = `W${a[1]}`;
  else if (AA3.has(a[3])) t = `${["KCX", "MSE", "SEP", "TPO", "PTR", "CSO", "LLP", "MLY", "HYP"].includes(a[3]) ? a[3] : title(a[3])}${a[1]}${a[2] || ""}`;
  else t = a[3];
  return many ? `${t} ${a[4]}` : t;
}
// Place labels next to their atoms where they cover the least: try 12 directions at three distances, avoid the image
// edge, other labels and the drawn atoms; the most crowded labels go first. A label placed far out gets a leader line.
function figureLabels(cam, W, H, fontPx, half_w) {
  const v = new THREE.Vector3(), pxA = W / (2 * half_w);
  const toPx = (p) => { v.copy(p).applyMatrix4(world.matrixWorld).project(cam); return [(v.x + 1) / 2 * W, (1 - v.y) / 2 * H, v.z]; };
  // occupancy of drawn atoms on a grid
  const cell = Math.max(4, fontPx / 3), gw = Math.ceil(W / cell), gh = Math.ceil(H / cell), occ = new Uint8Array(gw * gh);
  const mark = (x, y, r) => {
    for (let gy = Math.max(0, Math.floor((y - r) / cell)); gy <= Math.min(gh - 1, Math.floor((y + r) / cell)); gy++)
      for (let gx = Math.max(0, Math.floor((x - r) / cell)); gx <= Math.min(gw - 1, Math.floor((x + r) / cell)); gx++) occ[gy * gw + gx] = 1;
  };
  for (const mol of S.mols.values()) {
    if (!mol.group.visible) continue;
    for (let i = 0; i < mol.data.atoms.length; i++) {
      if (mol.zone && !mol.zone.has(i)) continue;
      if (mol.hideH && mol.data.element[i] === "H") continue;
      const [x, y, z] = toPx(P3(mol, i));
      if (z < -1 || z > 1 || x < 0 || y < 0 || x > W || y > H) continue;
      mark(x, y, 0.45 * pxA);
    }
  }
  const occFrac = (bx, by, bw, bh) => {
    let n = 0, t = 0;
    for (let gy = Math.max(0, Math.floor(by / cell)); gy <= Math.min(gh - 1, Math.floor((by + bh) / cell)); gy++)
      for (let gx = Math.max(0, Math.floor(bx / cell)); gx <= Math.min(gw - 1, Math.floor((bx + bw) / cell)); gx++) { t++; n += occ[gy * gw + gx]; }
    return t ? n / t : 0;
  };
  const meas = document.createElement("canvas").getContext("2d");
  meas.font = `${fontPx}px Helvetica, Arial, sans-serif`;
  const items = [];
  const perRes = new Map();
  for (const k of S.labels.keys()) {
    const [mid, idx] = k.split(":").map(Number), a = S.mols.get(mid)?.data.atoms[idx];
    if (a) perRes.set(`${mid}:${a[0]}/${a[1]}${a[2]}`, (perRes.get(`${mid}:${a[0]}/${a[1]}${a[2]}`) || 0) + 1);
  }
  for (const k of S.labels.keys()) {
    const [mid, idx] = k.split(":").map(Number), mol = S.mols.get(mid);
    if (!mol) continue;
    const [x, y, z] = toPx(P3(mol, idx));
    if (z < -1 || z > 1 || x < 0 || y < 0 || x > W || y > H) continue;
    const a = mol.data.atoms[idx];
    items.push({ kind: "label", text: paperLabel(mid, idx, perRes.get(`${mid}:${a[0]}/${a[1]}${a[2]}`) > 1), ax: x, ay: y });
  }
  for (const d of S.distances) {
    if (!d.label) continue;
    const A = S.mols.get(d.a.mol), B = S.mols.get(d.b.mol);
    if (!A || !B) continue;
    const pa = toPx(P3(A, d.a.idx)), pb = toPx(P3(B, d.b.idx));
    const nx = -(pb[1] - pa[1]), ny = pb[0] - pa[0], nl = Math.hypot(nx, ny) || 1;
    items.push({ kind: "distance", text: d.div.textContent, ax: (pa[0] + pb[0]) / 2, ay: (pa[1] + pb[1]) / 2, perp: [nx / nl, ny / nl] });
  }
  for (const it of items) { it.w = meas.measureText(it.text).width; it.h = fontPx * 1.05; it.crowd = occFrac(it.ax - 2 * fontPx, it.ay - 2 * fontPx, 4 * fontPx, 4 * fontPx); }
  items.sort((p, q) => q.crowd - p.crowd);
  const placed = [], result = [];
  const overlap = (b, c) => Math.max(0, Math.min(b[0] + b[2], c[0] + c[2]) - Math.max(b[0], c[0])) * Math.max(0, Math.min(b[1] + b[3], c[1] + c[3]) - Math.max(b[1], c[1]));
  const r0 = 0.5 * pxA + 0.25 * fontPx;
  for (const it of items) {
    let best = null;
    const dirs = it.kind === "distance" ? [it.perp, [-it.perp[0], -it.perp[1]]]
      : Array.from({ length: 12 }, (_, k) => [Math.cos(k * Math.PI / 6), -Math.sin(k * Math.PI / 6)]);
    [r0, r0 + 0.9 * fontPx, r0 + 2 * fontPx].forEach((r, ri) => {
      for (const [dx, dy] of dirs) {
        const cx = it.ax + dx * (r + it.w / 2 * Math.abs(dx)), cy = it.ay + dy * (r + it.h / 2 * Math.abs(dy));
        const box = [cx - it.w / 2, cy - it.h / 2, it.w, it.h];
        const outside = box[0] < 2 || box[1] < 2 || box[0] + box[2] > W - 2 || box[1] + box[3] > H - 2;
        let lab = 0;
        for (const p of placed) lab += overlap(box, p.box) / (it.w * it.h);
        const cost = (outside ? 1e3 : 0) + 40 * lab + 6 * occFrac(...box) + 0.8 * ri + (dx < -0.1 ? 0.15 : 0) + (dy > 0.1 ? 0.1 : 0);
        if (!best || cost < best.cost) best = { cost, box, r, dx, dy };
      }
    });
    const out = { kind: it.kind, text: it.text, x: best.box[0], y: best.box[1] + it.h / 2, w: it.w };
    if (best.r > r0 + 0.5 * fontPx && it.kind === "label") {       // leader line from the atom to the label's near edge
      const ex = Math.min(Math.max(it.ax, best.box[0]), best.box[0] + best.box[2]);
      const ey = Math.min(Math.max(it.ay, best.box[1]), best.box[1] + best.box[3]);
      const len = Math.hypot(ex - it.ax, ey - it.ay) || 1, s0 = 0.35 * pxA;
      out.leader = [it.ax + (ex - it.ax) * s0 / len, it.ay + (ey - it.ay) * s0 / len, ex, ey];
    }
    placed.push({ box: best.box });
    result.push(out);
  }
  return result;
}
function drawLabels(g, labels, fontPx, bg) {
  const col = bg === "black" ? "#f2f2f2" : "#111111";
  g.font = `${fontPx}px Helvetica, Arial, sans-serif`;
  g.textBaseline = "middle";
  g.textAlign = "left";
  g.strokeStyle = col; g.lineWidth = Math.max(1, fontPx / 14);
  for (const lb of labels) {
    if (lb.leader) { g.beginPath(); g.moveTo(lb.leader[0], lb.leader[1]); g.lineTo(lb.leader[2], lb.leader[3]); g.stroke(); }
    g.fillStyle = col;
    g.fillText(lb.text, lb.x, lb.y);
  }
}
async function renderFigure(o) {
  for (let t = 0; anim && t < 100; t++) await sleep(20);
  const W = o.width, H = o.height, bg = o.background;
  const maxDim = Math.min(renderer.capabilities.maxRenderbufferSize || 8192, 8192);
  const ss = Math.max(1, Math.min(2, Math.floor(maxDim / Math.max(W, H))));
  const RW = W * ss, RH = H * ss;
  const cam = new THREE.OrthographicCamera(-o.half_w, o.half_w, o.half_h, -o.half_h, Math.max(0.1, S.zoom - S.slab), S.zoom + S.slab);
  cam.position.set(0, 0, S.zoom);
  cam.updateProjectionMatrix();
  // ---- switch the scene to figure mode
  const restore = [];
  const swap = (obj, key, val) => { restore.push([obj, key, obj[key]]); obj[key] = val; };
  updateCamera(); scene.updateMatrixWorld();
  const bgColor = bg === "black" ? 0x000000 : 0xffffff;
  swap(scene, "background", bg === "transparent" ? null : new THREE.Color(bgColor));
  swap(scene.fog, "color", new THREE.Color(bgColor));
  swap(scene.fog, "near", S.zoom - S.slab * 0.2);
  swap(scene.fog, "far", S.zoom + S.slab * 1.6);
  swap(crossObj, "visible", false);
  for (const m of S.maps.values()) swap(m.obj, "visible", false);
  if (SLICE.mesh && o.style !== "slice") swap(SLICE.mesh, "visible", false);
  for (const d of S.distances) swap(d.line, "visible", false);
  if (PREVIEW.dist) swap(PREVIEW.dist, "visible", false);
  for (const mol of S.mols.values()) mol.group.traverse((x) => {
    if (x.isInstancedMesh) {
      if (x.geometry === sphereGeo) swap(x, "geometry", FIG.sphere);
      else if (x.geometry === cylGeo) swap(x, "geometry", FIG.cyl);
      if (x.material === solidMat) swap(x, "material", FIG.mat);
    }
  });
  const lineScale = RW / Math.max(1, renderer.domElement.clientWidth) * 0.6;
  for (const lm of lineMaterials) { swap(lm, "linewidth", lm.linewidth * lineScale); lm.resolution.set(RW, RH); }
  const distGrp = figureDistances(bg);
  const rt = new THREE.WebGLRenderTarget(RW, RH, { samples: 4 });
  const prevClear = renderer.getClearColor(new THREE.Color()), prevAlpha = renderer.getClearAlpha();
  const shoot = () => {
    if (o.style === "slice") updateSlice();
    renderer.setRenderTarget(rt);
    renderer.setClearColor(bgColor, bg === "transparent" ? 0 : 1);
    renderer.clear();
    renderer.render(scene, cam);
    const px = new Uint8Array(RW * RH * 4);
    renderer.readRenderTargetPixels(rt, 0, 0, RW, RH, px);
    renderer.setRenderTarget(null);
    const big = document.createElement("canvas");
    big.width = RW; big.height = RH;
    const img = new ImageData(RW, RH), row = RW * 4;
    for (let y = 0; y < RH; y++) img.data.set(px.subarray((RH - 1 - y) * row, (RH - y) * row), y * row);   // flip
    big.getContext("2d").putImageData(img, 0, 0);
    const out = document.createElement("canvas");
    out.width = W; out.height = H;
    const g = out.getContext("2d");
    g.imageSmoothingQuality = "high";
    g.drawImage(big, 0, 0, W, H);
    return out;
  };
  let figMaps = null, result;
  try {
    if (o.maps.length) { figMaps = await buildFigureMaps(o, true); world.add(figMaps); }
    const plain = shoot();
    const labels = figureLabels(cam, W, H, o.font_px, o.half_w);
    const lab = document.createElement("canvas");
    lab.width = W; lab.height = H;
    const g = lab.getContext("2d");
    g.drawImage(plain, 0, 0);
    drawLabels(g, labels, o.font_px, bg);
    result = { plain: plain.toDataURL("image/png"), png: lab.toDataURL("image/png"), labels, font_px: o.font_px,
      hbonds: S.distances.some((d) => d.style === "hbond"), coordination: [...S.mols.values()].some((m) => m.group.visible && m.hasCoord),
      carbon: Object.fromEntries([...S.mols.keys()].map((id) => [String(id), "#" + new THREE.Color(carbonColor(id)).getHexString()])) };
    if (o.uncarved) {
      disposeGroup(figMaps);
      figMaps = await buildFigureMaps(o, false);
      world.add(figMaps);
      const u = shoot();
      drawLabels(u.getContext("2d"), labels, o.font_px, bg);
      result.uncarved = u.toDataURL("image/png");
    }
  } finally {
    if (figMaps) disposeGroup(figMaps);
    disposeGroup(distGrp);
    rt.dispose();
    for (let k = restore.length - 1; k >= 0; k--) { const [obj, key, val] = restore[k]; obj[key] = val; }
    renderer.setClearColor(prevClear, prevAlpha);
    resize();
    redraw();
  }
  return result;
}

// ------------------------------------------------------------- figure preview
// The viewer as the figure will look: the figure look (cartoon, zone; applied on the server), white background,
// figure atoms, thick dashes, placed paper-style labels. Maps stay as mesh (they become surfaces in the figure).
const PREVIEW = { dist: null, radius: 5, timer: null, labelsEl: null };
function setPreview(on, radius) {
  if (radius) PREVIEW.radius = radius;
  const was = !!S.preview;
  S.preview = !!on;
  $("#btn-preview").setAttribute("aria-pressed", String(S.preview));
  $("#btn-preview").classList.toggle("active", S.preview);
  document.body.classList.toggle("preview", S.preview);
  const bg = S.preview ? 0xffffff : 0x000000;
  scene.background = new THREE.Color(bg);
  scene.fog.color = new THREE.Color(bg);
  if (was !== S.preview) for (const id of S.mols.keys()) rebuildMolGeometry(id);
  for (const d of S.distances) d.line.visible = !S.preview;
  previewDistances();
  if (!S.preview && PREVIEW.labelsEl) PREVIEW.labelsEl.innerHTML = "";
  status(S.preview ? "Figure preview: as the figure will look (maps become surfaces in the figure). Click atoms to label them; the shown residues follow." : "");
  redraw();
}
function previewDistances() {
  if (PREVIEW.dist) { disposeGroup(PREVIEW.dist); PREVIEW.dist = null; }
  for (const d of S.distances) d.line.visible = !S.preview;
  if (S.preview && S.distances.length) PREVIEW.dist = figureDistances("white");
}
function refreshPreviewSoon() {          // after the label click reached the server (pushView waits 150 ms)
  clearTimeout(PREVIEW.timer);
  PREVIEW.timer = setTimeout(() => cmd("figure_preview", { on: true, look_radius: PREVIEW.radius }).catch(() => {}), 450);
}
function updatePreviewLabels() {
  if (!PREVIEW.labelsEl) {
    PREVIEW.labelsEl = document.createElement("div");
    PREVIEW.labelsEl.id = "preview-labels";
    vp.appendChild(PREVIEW.labelsEl);
  }
  const W = renderer.domElement.clientWidth, H = renderer.domElement.clientHeight;
  if (!W || !H) return;
  const fontPx = 13, half_w = S.zoom * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * camera.aspect;
  scene.updateMatrixWorld();
  const labels = figureLabels(camera, W, H, fontPx, half_w);
  let svg = `<svg width="${W}" height="${H}">`, html = "";
  for (const lb of labels) {
    if (lb.leader) svg += `<line x1="${lb.leader[0]}" y1="${lb.leader[1]}" x2="${lb.leader[2]}" y2="${lb.leader[3]}"/>`;
    html += `<span style="left:${lb.x}px;top:${lb.y}px"></span>`;
  }
  PREVIEW.labelsEl.innerHTML = svg + "</svg>" + html;
  PREVIEW.labelsEl.querySelectorAll("span").forEach((sp, k) => { sp.textContent = labels[k].text; });
}
$("#btn-preview").onclick = () => cmd("figure_preview", { on: !S.preview, look_radius: PREVIEW.radius });

// ------------------------------------------------------------------- map slices
// As in shared pages: the map drawn on a plane (green +, red -; the map colour for density maps; brighter towards
// 3 x level, contour lines at 1, 2, 4 ... x level). The plane faces the viewer through the centre and follows the view,
// or is fixed in the molecule (lock). The server sends the map values in sigma on a cube around the plane; the shader
// samples them (thickness: the strongest value within +/- thickness/2). State is kept on the server (`slice` command).
const SLICE = { style: "mesh", map: null, lock: null, depth: 0, thickness: 0, mesh: null, box: null, busy: false,
  fresh: false, failed: false };
const SLICE_VS = `
uniform mat4 uPlane;
varying vec3 vModel;
varying vec2 vXY;
void main() {
  vModel = (uPlane * vec4(position, 1.0)).xyz;
  vXY = position.xy;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;
const SLICE_FS = `
precision highp float;
precision highp sampler3D;
uniform sampler3D uMap;
uniform mat3 uGinv;
uniform vec3 uOrigin, uN, uPos, uNeg, uNormal;
uniform float uLevel, uThick;
uniform int uSigned, uWrap, uEmpty;
varying vec3 vModel;
varying vec2 vXY;
// a map stored on a block or box (not the periodic cell) has no values outside it
bool outside(vec3 idx) {
  return uWrap == 0 && (any(lessThan(idx, vec3(-0.5))) || any(greaterThan(idx, uN - 0.5)));
}
float sampleAt(vec3 p) {
  vec3 idx = uGinv * (p - uOrigin);
  return outside(idx) ? 0.0 : texture(uMap, (idx.zyx + 0.5) / uN.zyx).r;
}
void main() {
  if (uEmpty == 1) { gl_FragColor = vec4(0.0, 0.0, 0.0, 0.45); return; }   // no map shown: the bare plane
  if (outside(uGinv * (vModel - uOrigin))) discard;
  // thin plane: the value on it; thick slice: the strongest value (sign kept) within +/- uThick / 2 of it
  float v = sampleAt(vModel);
  if (uThick > 0.01) {
    for (int s = 0; s < 13; s++) {
      float w = sampleAt(vModel + uNormal * uThick * (float(s) / 12.0 - 0.5));
      if (uSigned == 0 ? w > v : abs(w) > abs(v)) v = w;
    }
  }
  float a = (uSigned == 0 && v < 0.0) ? 0.0 : abs(v);
  vec3 hue = v >= 0.0 ? uPos : uNeg;
  float k = clamp(a / (3.0 * uLevel), 0.0, 1.0);
  vec3 col = hue * k;
  float alpha = 0.45 + 0.5 * k;
  float g = log2(max(a, 1e-6) / uLevel);
  float line = (1.0 - smoothstep(0.0, 1.5 * fwidth(g), abs(fract(g + 0.5) - 0.5))) * smoothstep(0.85, 0.95, a / uLevel);
  col = mix(col, min(hue * 1.2 + 0.3, vec3(1.0)), line);
  alpha = max(alpha, line);
  gl_FragColor = vec4(col * alpha, alpha);
}`;

function makePlaneMesh(outlineColor) {
  const mat = new THREE.ShaderMaterial({
    vertexShader: SLICE_VS, fragmentShader: SLICE_FS, side: THREE.DoubleSide, transparent: true,
    premultipliedAlpha: true, depthWrite: false,
    uniforms: {
      uMap: { value: null }, uPlane: { value: new THREE.Matrix4() }, uGinv: { value: new THREE.Matrix3() },
      uOrigin: { value: new THREE.Vector3() }, uN: { value: new THREE.Vector3(1, 1, 1) }, uPos: { value: new THREE.Color() },
      uNeg: { value: new THREE.Color() }, uLevel: { value: 1 }, uSigned: { value: 1 }, uWrap: { value: 0 }, uEmpty: { value: 1 },
      uNormal: { value: new THREE.Vector3(0, 0, 1) }, uThick: { value: 0 },
    },
  });
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), mat);
  mesh.matrixAutoUpdate = false;
  mesh.frustumCulled = false;
  mesh.renderOrder = 5;
  const frame = new LineSegmentsGeometry();
  frame.setPositions([-1, -1, 0, 1, -1, 0, 1, -1, 0, 1, 1, 0, 1, 1, 0, -1, 1, 0, -1, 1, 0, -1, -1, 0]);
  const outline = new LineSegments2(frame, lineMat(2, { color: outlineColor, vertexColors: false }));
  outline.frustumCulled = false;
  mesh.add(outline);
  world.add(mesh);
  return mesh;
}
renderer.debug.onShaderError = (gl, program, vs, fs) => {
  SLICE.failed = true;
  console.error("shader error", gl.getProgramInfoLog(program), gl.getShaderInfoLog(fs));
  status("The map slice is not available in this browser; back to mesh.");
  setTimeout(() => setSliceStyle("mesh"), 0);
};

const sliceMap = () => S.maps.get(SLICE.map) || [...S.maps.values()].find((m) => m.info.visible) || null;
const sl_M = new THREE.Matrix4(), sl_T = new THREE.Matrix4(), sl_Rt = new THREE.Matrix4(), sl_q = new THREE.Quaternion();
const sl_n = new THREE.Vector3(), sl_p0 = new THREE.Vector3(), sl_c = new THREE.Vector3();
function slicePlacement(r) {
  let c = S.center;
  if (SLICE.lock) { c = sl_c.set(...SLICE.lock.center); sl_Rt.makeRotationFromQuaternion(sl_q.set(...SLICE.lock.orient)).transpose(); }
  else sl_Rt.makeRotationFromQuaternion(rot.quaternion).transpose();     // columns: the view axes in model coordinates
  sl_M.makeTranslation(c.x, c.y, c.z).multiply(sl_Rt).multiply(sl_T.makeTranslation(0, 0, SLICE.depth)).multiply(sl_T.makeScale(r, r, 1));
  sl_n.setFromMatrixColumn(sl_Rt, 2);
  sl_p0.copy(c).addScaledVector(sl_n, SLICE.depth);
}
async function fetchBox(m, p0, r) {
  SLICE.busy = true;
  try {
    const resp = await fetch(`/api/map/${m.info.id}/box?x=${p0.x}&y=${p0.y}&z=${p0.z}&r=${r}`);
    if (!resp.ok) { status(`⚠ slice of map ${m.info.id}: ${(await resp.text()).slice(0, 200)}`); return; }
    const buf = await resp.arrayBuffer();
    const hlen = new DataView(buf).getUint32(0, true);
    const h = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hlen)));
    const vals = new Float32Array(buf, 4 + hlen);
    const half = new Uint16Array(vals.length);
    for (let q = 0; q < vals.length; q++) half[q] = THREE.DataUtils.toHalfFloat(vals[q]);
    const [n0, n1, n2] = h.n;
    const tex = new THREE.Data3DTexture(half, n2, n1, n0);    // x = third index (fastest), z = first
    tex.format = THREE.RedFormat;
    tex.type = THREE.HalfFloatType;
    tex.minFilter = tex.magFilter = THREE.LinearFilter;
    tex.wrapS = tex.wrapT = tex.wrapR = THREE.ClampToEdgeWrapping;
    tex.unpackAlignment = 1;
    tex.needsUpdate = true;
    if (SLICE.box) SLICE.box.tex.dispose();
    SLICE.box = { map: m.info.id, center: p0.clone(), r, tex, origin: h.origin, step: h.step, n: h.n };
    if (h.max_sigma != null && m.info.max_sigma == null) { m.info.max_sigma = h.max_sigma; updateLevelUI(m); }
  } finally { SLICE.busy = false; redraw(); }
}
// per frame: place the plane; fetch the map around it when it moved out of the stored cube
function updateSlice() {
  if (SLICE.style !== "slice") { if (SLICE.mesh) SLICE.mesh.visible = false; SLICE.fresh = true; return; }
  if (!SLICE.mesh) SLICE.mesh = makePlaneMesh(0xe6c229);
  const m = sliceMap(), r = m ? m.info.radius : 10;
  slicePlacement(r);
  const mesh = SLICE.mesh, u = mesh.material.uniforms;
  mesh.matrix.copy(sl_M);
  mesh.matrixWorldNeedsUpdate = true;
  mesh.visible = true;
  u.uPlane.value.copy(sl_M);
  u.uNormal.value.copy(sl_n);
  u.uThick.value = SLICE.thickness;
  if (!m) { u.uEmpty.value = 1; SLICE.fresh = true; return; }
  const need = r * Math.SQRT2 + SLICE.thickness / 2 + 0.5, b = SLICE.box;
  SLICE.fresh = !!b && b.map === m.info.id && b.r >= need && b.center.distanceTo(sl_p0) <= b.r - need;
  if (!SLICE.fresh && !SLICE.busy) fetchBox(m, sl_p0.clone(), Math.min(40, need + 3));
  const have = b && b.map === m.info.id;
  u.uEmpty.value = have ? 0 : 1;
  if (have) {
    u.uMap.value = b.tex;
    u.uGinv.value.set(1 / b.step, 0, 0, 0, 1 / b.step, 0, 0, 0, 1 / b.step);
    u.uOrigin.value.set(...b.origin);
    u.uN.value.set(...b.n);
  }
  const diff = m.info.kind === "diff";
  u.uPos.value.set(diff ? "#00d000" : m.info.color);
  u.uNeg.value.set("#e00000");
  u.uSigned.value = diff ? 1 : 0;
  u.uLevel.value = m.info.level_sigma;
}
function sliceReady() { render1(); return SLICE.fresh && !SLICE.busy; }

function sendSlice() {
  send({ type: "slice", style: SLICE.style, map: SLICE.map, lock: SLICE.lock, depth: SLICE.depth, thickness: SLICE.thickness });
}
function renderSliceUI() {
  document.querySelectorAll("#style-row button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.style === SLICE.style)));
  $("#slice-box").classList.toggle("hidden", SLICE.style !== "slice");
  $("#slice-depth").value = String(SLICE.depth);
  $("#slice-depth-out").textContent = `${SLICE.depth >= 0 ? "+" : "−"}${Math.abs(SLICE.depth).toFixed(1)} Å`;
  $("#slice-thick").value = String(SLICE.thickness);
  $("#slice-thick-out").textContent = SLICE.thickness > 0.01 ? `${SLICE.thickness.toFixed(1)} Å` : "thin";
  const lk = $("#slice-lock");
  lk.setAttribute("aria-pressed", String(!!SLICE.lock));
  lk.textContent = SLICE.lock ? "Release" : "Fix plane";
}
function renderSliceMapUI() { redraw(); }
function applySlice(m) {        // from the server (command, restored view, or our own change echoed)
  const was = SLICE.style;
  for (const k of ["style", "map", "lock", "depth", "thickness"]) if (k in m) SLICE[k] = m[k];
  if (SLICE.style === "slice" && SLICE.failed) SLICE.style = "mesh";
  renderSliceUI();
  if (was !== SLICE.style) { applyMapVisibility(); for (const x of S.maps.values()) x.lastCenter = null; scheduleContour(true); }
  redraw();
}
function setSliceStyle(style) {
  applySlice({ style });
  sendSlice();
  if (style === "slice") status("Slice: the map on a plane through the centre. Fix plane locks it to the molecule.");
}
document.querySelectorAll("#style-row button").forEach((b) => b.addEventListener("click", () => setSliceStyle(b.dataset.style)));
$("#slice-depth").addEventListener("input", (e) => { applySlice({ depth: +e.target.value }); sendSlice(); });
$("#slice-thick").addEventListener("input", (e) => { applySlice({ thickness: +e.target.value }); sendSlice(); });
$("#slice-lock").addEventListener("click", () => {
  applySlice({ lock: SLICE.lock ? null : { center: S.center.toArray(), orient: rot.quaternion.toArray() } });
  sendSlice();
  if (SLICE.lock) status("Plane fixed in the molecule: rotate to see it from the side; Depth moves it along its normal.");
});
$("#slice-face").addEventListener("click", () => { applySlice({ lock: null, depth: 0 }); sendSlice(); });
$("#slice-atoms").addEventListener("click", async () => {
  const atoms = S.selection.map((p) => `${p.mol}#${atomLabel(p.mol, p.idx)}`);
  if (atoms.length < 2) { status("Click two or more atoms (they get labels) for a plane through them."); return; }
  try { await cmd("slice", { atoms }); } catch (err) { status(`⚠ ${err.message || err}`); }
});

// ------------------------------------------------------------------- start
async function init() {
  resize();
  const st = await fetch("/api/state").then((r) => r.json());
  for (const m of st.molecules) setMolecule(m);
  S.activeMol = st.active_mol;
  setMaps(st.maps, st.refinement_map);
  if (st.view?.center) S.center.set(...st.view.center);
  for (const l of st.log) { $("#log").textContent += l + "\n"; }
  for (const c of st.chat) chatAdd(c);
  $("#chat-bash").checked = !!st.chat_settings?.allow_bash;
  if (st.rsr_settings) setRsrSettings(st.rsr_settings);
  if (st.rsr) startRSR({ mol: st.rsr.mol, idx: st.rsr.idx });
  if (st.tour) renderTour(st.tour);
  if (st.slice) applySlice(st.slice);
  if (st.figure_preview) setPreview(true);
  renderMolList();
  connect();
  setTimeout(pushView, 500);   // the server learns this tab's zoom, slab and rotation
  scheduleContour(true);
  render();
  new ResizeObserver(() => { resize(); redraw(); }).observe(vp);
}

// Drag the splitter to resize the side panel; width is remembered per browser.
(function sidePanelSplitter() {
  const side = $("#side"), split = $("#splitter"), KEY = "moorlight.sideWidth";
  const setW = (w) => {
    w = Math.max(260, Math.min(w, window.innerWidth - 200));
    side.style.width = w + "px";
    return w;
  };
  try { const w = +localStorage.getItem(KEY); if (w) setW(w); } catch {}
  split.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    split.setPointerCapture(e.pointerId);
    split.classList.add("dragging"); document.body.classList.add("col-resizing");
    const x0 = e.clientX, w0 = side.getBoundingClientRect().width;
    const move = (ev) => setW(w0 + x0 - ev.clientX);
    const up = () => {
      split.removeEventListener("pointermove", move);
      split.classList.remove("dragging"); document.body.classList.remove("col-resizing");
      try { localStorage.setItem(KEY, Math.round(side.getBoundingClientRect().width)); } catch {}
    };
    split.addEventListener("pointermove", move);
    split.addEventListener("pointerup", up, { once: true });
    split.addEventListener("pointercancel", up, { once: true });
  });
  split.addEventListener("dblclick", () => {
    side.style.width = "";
    try { localStorage.removeItem(KEY); } catch {}
  });
})();
init();
