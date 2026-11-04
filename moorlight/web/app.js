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

function resize() {
  const w = vp.clientWidth, h = vp.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  for (const m of lineMaterials) m.resolution.set(w * devicePixelRatio, h * devicePixelRatio);
}
window.addEventListener("resize", resize);

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
  crossObj.visible = S.cross;
  renderer.render(scene, camera);
  updateLabels();
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
    send({ type: "view", center: S.center.toArray().map((v) => +v.toFixed(2)), zoom: S.zoom,
      selection: sel, clicked: S.clicked ? atomLabel(S.clicked.mol, S.clicked.idx) : null,
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
    case "maps_changed": for (const mm of S.maps.values()) mm.lastCenter = null; scheduleContour(true); break;
    case "view_cmd": if (m.center) moveTo(new THREE.Vector3(...m.center)); break;
    case "log": addLog(m.text); break;
    case "rsr_start": startRSR(m); renderUndo(); break;
    case "rsr_coords": rsrCoords(m); break;
    case "rsr_status": rsrStatus(m); break;
    case "rsr_end": endRSR(); renderUndo(); break;
    case "screenshot_request": needsRender = true; render1(); send({ type: "reply", id: m.id, data: renderer.domElement.toDataURL("image/png") }); break;
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
function render1() { updateCamera(); renderer.render(scene, camera); }

// --------------------------------------------------------------- molecules
function carbonColor(molId) {
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
  mol.group.visible = data.visible;
  rebuildMolGeometry(data.id);
  // drop labels pointing at atoms that no longer exist
  for (const [k, div] of S.labels) {
    const [mid, idx] = k.split(":").map(Number);
    if (mid === data.id && idx >= data.atoms.length) { div.remove(); S.labels.delete(k); }
  }
  renderMolList();
  redraw();
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
  for (const ch of [...mol.group.children]) { mol.group.remove(ch); ch.geometry.dispose(); }
  const isH = (i) => d.element[i] === "H";
  const keep = [];  // bond indices to draw
  for (let k = 0; k < d.bonds.length / 2; k++) {
    if (mol.hideH && (isH(d.bonds[2*k]) || isH(d.bonds[2*k+1]))) continue;
    keep.push(k);
  }
  const bonded = new Uint8Array(d.atoms.length);
  for (const i of d.bonds) bonded[i] = 1;
  const lone = [];
  for (let i = 0; i < d.atoms.length; i++) if (!bonded[i] && !(mol.hideH && isH(i))) lone.push(i);
  if (mol.style === "lines") {
    const bonds = [];
    for (const k of keep) bonds.push(d.bonds[2*k], d.bonds[2*k+1]);
    const b = bondGeometry(mol.xyz, bonds, d.element, id);
    mol.group.add(makeLines(b.pos, b.col, 3));
    const isWater = (i) => ["HOH", "WAT", "DOD"].includes(d.atoms[i][3]);
    const cr = crossGeometry(mol.xyz, lone, d.element, id, (i) => (isWater(i) ? 0.3 : 0.55));
    if (cr.pos.length) mol.group.add(makeLines(cr.pos, cr.col, 2));
  } else {
    for (const o of stickMeshes(mol, id, keep, lone)) mol.group.add(o);
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

// ---------------------------------------------------------- ball & stick
const STICK_R = 0.13, BALL_R = 0.24, H_STICK_R = 0.07, H_BALL_R = 0.12, WATER_R = 0.28, ION_R = 0.6;
const sphereGeo = new THREE.SphereGeometry(1, 12, 8);
const cylGeo = new THREE.CylinderGeometry(1, 1, 1, 10, 1, true);
const solidMat = new THREE.MeshLambertMaterial({ color: 0xffffff });
const _m = new THREE.Matrix4(), _q = new THREE.Quaternion(), _v = new THREE.Vector3(), _s = new THREE.Vector3();
const _up = new THREE.Vector3(0, 1, 0), _c = new THREE.Color();

function stickMeshes(mol, id, keep, lone) {
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
    const im = new THREE.InstancedMesh(cylGeo, solidMat, cyl.length);
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
  for (let i = 0; i < d.atoms.length; i++) if (!(mol.hideH && isH(i))) allBalls.push(i);
  if (allBalls.length) {
    const im = new THREE.InstancedMesh(sphereGeo, solidMat, allBalls.length);
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
    m.obj.visible = info.visible;
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
  renderMapList();
  scheduleContour(true);
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
      if (!m.info.visible) continue;
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
  const buf = await fetch(url).then((r) => r.arrayBuffer());
  const hlen = new DataView(buf).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hlen)));
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
    d.div.style.top = `${(1 - (pa.y + pb.y) / 2) / 2 * rect.height}px`;
  }
}
function clearLabels() {
  for (const div of S.labels.values()) div.remove();
  S.labels.clear();
  for (const d of S.distances) { d.div.remove(); overlay.remove(d.line); }
  S.distances = [];
  redraw();
}
function addDistance(a, b) {
  const A = S.mols.get(a.mol).xyz, B = S.mols.get(b.mol).xyz;
  const pa = new THREE.Vector3(A[3*a.idx], A[3*a.idx+1], A[3*a.idx+2]);
  const pb = new THREE.Vector3(B[3*b.idx], B[3*b.idx+1], B[3*b.idx+2]);
  const g = new THREE.BufferGeometry().setFromPoints([pa, pb]);
  const line = new THREE.LineSegments(g, new THREE.LineDashedMaterial({ color: 0x99ffff, dashSize: 0.2, gapSize: 0.15 }));
  line.computeLineDistances();
  overlay.add(line);
  const div = document.createElement("div");
  div.className = "label dist"; div.textContent = `${pa.distanceTo(pb).toFixed(2)} Å`;
  $("#labels").appendChild(div);
  S.distances.push({ a, b, line, div });
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

function changeLevel(m, step) {
  m.info.level_sigma = Math.max(0.1, +(m.info.level_sigma + step).toFixed(2));
  m.lastLevel = null;
  renderMapList();
  status(`${m.info.name}: ${m.info.level_sigma.toFixed(2)}σ (${(m.info.level_sigma * m.info.rms).toFixed(3)} e/Å³)`);
  scheduleContour(); pushView();
}

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
      if (S.toolClicks.length === 2) { addDistance(S.toolClicks[0], S.toolClicks[1]); S.toolClicks = []; }
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
    else if (k === "d" || k === "D") { S.slab = Math.max(1, S.slab * 0.85); redraw(); }
    else if (k === "f" || k === "F") { S.slab = Math.min(200, S.slab * 1.15); redraw(); }
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
      `<button title="Toggle ball-and-stick / lines">${mol.style === "lines" ? "lines" : "sticks"}</button>` +
      `<button title="Make active (edits apply here)">edit</button><button title="Close">×</button>`;
    const [cb, hcb] = div.querySelectorAll("input");
    const styleBtn = div.querySelector("button[title^='Toggle ball']");
    styleBtn.onclick = () => { mol.style = mol.style === "lines" ? "sticks" : "lines"; rebuildMolGeometry(id); renderMolList(); redraw(); };
    if (hcb) hcb.onchange = () => { mol.hideH = !hcb.checked; rebuildMolGeometry(id); redraw(); };
    const act = div.querySelector("button[title^='Make active']"), close = div.querySelector("button[title='Close']");
    cb.onchange = () => { mol.group.visible = cb.checked; redraw(); };
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
    const div = document.createElement("div");
    div.className = "item";
    const color = i.kind === "diff" ? "linear-gradient(90deg,#00d000 50%,#e00000 50%)" : i.color;
    div.innerHTML = `<input type="radio" name="scroll" title="scroll wheel changes this map" ${S.scrollMap === id ? "checked" : ""}>` +
      `<input type="checkbox" ${i.visible ? "checked" : ""}>` +
      `<span class="swatch" style="background:${color}"></span>` +
      `<span class="name" title="${i.source}: ${i.f} / ${i.phi}">[${id}] ${i.name}</span>` +
      `<input type="number" step="0.1" value="${i.level_sigma.toFixed(2)}" title="contour level (σ)">` +
      `<span class="abs">${(i.level_sigma * i.rms).toFixed(3)}</span>` +
      `<span title="refinement map (target: map)" style="cursor:pointer;color:${S.refinementMap === id ? "#e6c229" : "#444"}">R</span>` +
      (i.kind === "diff"
        ? `<span title="difference map (target: difference)" style="cursor:pointer;color:${S.rsrSettings?.diff_map === id ? "#4fd0ff" : "#444"}">Δ</span>`
        : `<span style="width:0.7em"></span>`) +
      `<button title="Close">×</button>`;
    div.querySelector("button[title='Close']").onclick = () => cmd("close_map", { map_id: id });
    const [radio, cb, num] = div.querySelectorAll("input");
    const rbtn = div.querySelector("span[title^='refinement map']");
    const dbtn = div.querySelector("span[title^='difference map']");
    if (dbtn) dbtn.onclick = () => cmd("set_refinement_options", { diff_map: id });
    radio.onchange = () => { S.scrollMap = id; pushView(); };
    cb.onchange = () => { i.visible = cb.checked; m.obj.visible = cb.checked; m.lastCenter = null; scheduleContour(true); pushView(); redraw(); };
    num.onchange = () => { i.level_sigma = +num.value; m.lastLevel = null; scheduleContour(true); pushView(); renderMapList(); };
    rbtn.onclick = () => cmd("set_map", { map_id: id, refinement_map: true });
    box.appendChild(div);
    const o = document.createElement("option");
    o.value = id; o.textContent = `[${id}] ${i.name}`;
    sel.appendChild(o);
  }
  if (prev) sel.value = prev;
  else { const d = [...S.maps.values()].find((m) => m.info.kind === "diff" && m.info.visible); if (d) sel.value = d.info.id; }
}

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
  renderMolList();
  connect();
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
