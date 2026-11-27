// Moorlight share page: a read-only viewer for a snapshot written by `moorlight share`.
// Models, map boxes and preset planes are embedded in the page, and the build puts three.js in front of this
// script; maps are contoured here (marching cubes), so neither a server nor the network is needed. Drawing follows the main viewer (app.js): Coot colours, sticks, chicken wire.
import * as THREE from "three";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { edgeTable, triTable } from "three/addons/objects/MarchingCubes.js";

const $ = (s) => document.querySelector(s);
const DATA = JSON.parse($("#ml-data").textContent);
const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
const ELEMENT_COLORS = { C: 0xe6c229, N: 0x3060ff, O: 0xff2020, S: 0xe0e000, P: 0xff8000, H: 0xdddddd,
  SE: 0xe0a000, FE: 0xe07000, MG: 0x20e020, ZN: 0x8080c0, CA: 0x40c040, NA: 0x8040ff, CL: 0x20f020, K: 0x8040c0,
  MN: 0x9c7ac7, PD: 0x7fc8d8, PT: 0xd0d0e0, CU: 0xc88033, NI: 0x50d050, CO: 0xf090a0, BR: 0xa62929, I: 0x940094,
  F: 0x90e050 };
const CARBON_BY_MOL = [0xe6c229, 0x60c0ff, 0x80e080, 0xff80c0, 0xc0a0ff, 0xffffff];
const CONTOUR = { pos: 0x00d000, neg: 0xe00000, map: 0x3d7fff };
const WATERS = ["HOH", "WAT", "DOD"];

const S = {
  mols: [],            // {data, group, xyz, hideH, res (residue id per atom), hidden (per atom, Hide in front)}
  maps: [],            // {info, values: Float32Array, visible, level, obj, drawn, vol}
  center: new THREE.Vector3(),
  zoom: 30, slab: 9,
  stop: -1,
  labels: new Map(),   // "mol:idx" -> div
  distances: [],
  radius: DATA.contour_radius || 10,
  style: "mesh",       // mesh | slice
  planes: [],          // saved slice planes
  depth: 0,            // where the visible slab sits along the view axis, Å from the rotation centre (+: towards you)
};

// ------------------------------------------------------------------ three
const vp = $("#viewport");
const renderer = new THREE.WebGLRenderer({ antialias: true });
const DPR = Math.min(window.devicePixelRatio || 1, 2);
renderer.setPixelRatio(DPR);
vp.prepend(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);
scene.fog = new THREE.Fog(0x000000, 20, 40);
// Orthographic camera far from the focus: zooming only magnifies (S.zoom sets the visible height as a 30° view at
// that distance would), and depth is only the slab around the focus, moved with the depth bar.
const FOV = 30, CAM_DIST = 500, TAN_HALF = Math.tan(THREE.MathUtils.degToRad(FOV / 2));
const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 1, 1000);
let aspect = 1;
const rot = new THREE.Group();     // rotation about the centre
const world = new THREE.Group();   // translation (-centre)
rot.add(world); scene.add(rot);
const overlay = new THREE.Group(); world.add(overlay);
scene.add(new THREE.AmbientLight(0xffffff, 1.1));
const keyLight = new THREE.DirectionalLight(0xffffff, 2.0);
keyLight.position.set(0.4, 0.6, 1);
scene.add(keyLight);
const lineMaterials = new Set();

function lineMat(width, opts = {}) {
  const m = new LineMaterial({ linewidth: Math.max(1, width * DPR), worldUnits: false, fog: true, ...opts });
  m.resolution.set(vp.clientWidth * DPR, vp.clientHeight * DPR);
  lineMaterials.add(m);
  return m;
}

function resize() {
  const w = Math.max(1, vp.clientWidth), h = Math.max(1, vp.clientHeight);
  renderer.setSize(w, h, false);
  aspect = w / h;
  updateCamera();
  for (const m of lineMaterials) m.resolution.set(w * DPR, h * DPR);
  redraw();
}
new ResizeObserver(resize).observe(vp);

function updateCamera() {
  const h = S.zoom * TAN_HALF;              // half the visible height at the focus, in Å
  camera.left = -h * aspect; camera.right = h * aspect; camera.top = h; camera.bottom = -h;
  camera.position.set(0, 0, CAM_DIST);
  camera.near = CAM_DIST - S.depth - S.slab;
  camera.far = CAM_DIST - S.depth + S.slab;
  camera.updateProjectionMatrix();
  scene.fog.near = CAM_DIST - S.depth - S.slab * 0.3;
  scene.fog.far = CAM_DIST - S.depth + S.slab * 1.05;
  world.position.copy(S.center).multiplyScalar(-1);
}

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
  animate();
  if (!needsRender) return;
  needsRender = false;
  updateCamera();
  updateClear();
  sliceFrame();
  renderer.render(scene, camera);
  updateDepthBar();
  updateLabels();
}
const redraw = () => (needsRender = true);

function status(text) { $("#status").textContent = text; }

// --------------------------------------------------------------- molecules
function carbonColor(k) {
  const own = S.mols[k]?.data.color;
  return own ? new THREE.Color(own).getHex() : CARBON_BY_MOL[k % CARBON_BY_MOL.length];
}
function elColor(el, k) {
  const mono = S.mols[k]?.data.mono;
  if (mono) return carbonColor(k);
  el = (el || "").toUpperCase();
  if (el === "C") return carbonColor(k);
  return ELEMENT_COLORS[el] ?? 0xff40ff;
}

const STICK_R = 0.13, BALL_R = 0.24, H_STICK_R = 0.07, H_BALL_R = 0.12, WATER_R = 0.28, ION_R = 0.6;
const sphereGeo = new THREE.SphereGeometry(1, 12, 8);
const cylGeo = new THREE.CylinderGeometry(1, 1, 1, 10, 1, true);
const solidMat = new THREE.MeshLambertMaterial({ color: 0xffffff });
const _m = new THREE.Matrix4(), _q = new THREE.Quaternion(), _v = new THREE.Vector3(), _s = new THREE.Vector3();
const _up = new THREE.Vector3(0, 1, 0), _c = new THREE.Color();

function buildMolecule(k) {
  const mol = S.mols[k], d = mol.data;
  for (const ch of [...mol.group.children]) { mol.group.remove(ch); ch.geometry.dispose(); }
  mol.group.visible = d.visible !== false;
  const isH = (i) => d.element[i] === "H";
  const keep = [];
  for (let b = 0; b < d.bonds.length / 2; b++) {
    if (mol.hideH && (isH(d.bonds[2*b]) || isH(d.bonds[2*b+1]))) continue;
    keep.push(b);
  }
  const bonded = new Uint8Array(d.atoms.length);
  for (const i of d.bonds) bonded[i] = 1;
  const lone = [];
  for (let i = 0; i < d.atoms.length; i++) if (!bonded[i] && !(mol.hideH && isH(i))) lone.push(i);
  if (d.style === "lines") {
    const pos = [], col = [], seg = [];
    for (const b of keep) {
      const i = d.bonds[2*b], j = d.bonds[2*b+1], x = mol.xyz;
      const mx = (x[3*i] + x[3*j]) / 2, my = (x[3*i+1] + x[3*j+1]) / 2, mz = (x[3*i+2] + x[3*j+2]) / 2;
      pos.push(x[3*i], x[3*i+1], x[3*i+2], mx, my, mz, mx, my, mz, x[3*j], x[3*j+1], x[3*j+2]);
      _c.setHex(elColor(d.element[i], k)); col.push(_c.r, _c.g, _c.b, _c.r, _c.g, _c.b);
      _c.setHex(elColor(d.element[j], k)); col.push(_c.r, _c.g, _c.b, _c.r, _c.g, _c.b);
      seg.push(i, j);
    }
    if (pos.length) {
      const g = new LineSegmentsGeometry(); g.setPositions(pos); g.setColors(col);
      const ls = new LineSegments2(g, lineMat(2.5, { vertexColors: true }));
      ls.userData.lines = { pos: Float32Array.from(pos), col: Float32Array.from(col), atom: Int32Array.from(seg), shown: seg.length };
      mol.group.add(ls);
    }
    lone.length && mol.group.add(...stickMeshes(mol, k, [], lone));
  } else {
    mol.group.add(...stickMeshes(mol, k, keep, lone));
  }
  applyHidden(k);
  redraw();
}
// residues hidden by Hide in front: their sticks and balls shrink to nothing, their lines are left out
function applyHidden(k) {
  const mol = S.mols[k], hid = mol.hidden;
  for (const ch of mol.group.children) {
    const u = ch.userData;
    if (u.m0) {
      const arr = ch.instanceMatrix.array;
      for (let n = 0; n < u.atom.length; n++) {
        const o = 16 * n;
        if (hid[u.atom[n]]) arr.fill(0, o, o + 12);
        else arr.set(u.m0.subarray(o, o + 12), o);
      }
      ch.instanceMatrix.needsUpdate = true;
    } else if (u.lines) {
      const { pos, col, atom } = u.lines;
      const keep = [];
      for (let s = 0; s < atom.length; s++) if (!hid[atom[s]]) keep.push(s);
      if (keep.length === atom.length && u.lines.shown === atom.length) continue;
      const p = new Float32Array(6 * keep.length), q = new Float32Array(6 * keep.length);
      keep.forEach((s, n) => { p.set(pos.subarray(6*s, 6*s + 6), 6*n); q.set(col.subarray(6*s, 6*s + 6), 6*n); });
      ch.geometry.dispose();
      ch.geometry = new LineSegmentsGeometry();
      if (keep.length) { ch.geometry.setPositions(p); ch.geometry.setColors(q); }
      ch.visible = keep.length > 0;
      u.lines.shown = keep.length;
    }
  }
}

function stickMeshes(mol, k, keep, lone) {
  const d = mol.data, x = mol.xyz;
  const P = (i) => new THREE.Vector3(x[3*i], x[3*i+1], x[3*i+2]);
  const isH = (i) => d.element[i] === "H";
  const isWater = (i) => WATERS.includes(d.atoms[i][3]);
  const nb = new Map();
  for (const b of keep) {
    const i = d.bonds[2*b], j = d.bonds[2*b+1];
    if (isH(i) || isH(j)) continue;
    (nb.get(i) || nb.set(i, []).get(i)).push(j);
    (nb.get(j) || nb.set(j, []).get(j)).push(i);
  }
  const cyl = [];
  for (const b of keep) {
    const i = d.bonds[2*b], j = d.bonds[2*b+1];
    const pi = P(i), pj = P(j), mid = pi.clone().add(pj).multiplyScalar(0.5);
    const r = isH(i) || isH(j) ? H_STICK_R : STICK_R;
    const ci = elColor(d.element[i], k), cj = elColor(d.element[j], k);
    cyl.push([pi, mid, r, ci, i], [mid, pj, r, cj, j]);
    const order = d.orders ? d.orders[b] : 1;
    if (order === 2 || order === 3 || order === 4) {
      // second line offset towards a neighbour (inside of rings), shortened at both ends, like Coot
      const axis = pj.clone().sub(pi);
      const c = (nb.get(i) || []).find((n) => n !== j) ?? (nb.get(j) || []).find((n) => n !== i);
      let off;
      if (c !== undefined) {
        const ref = (nb.get(i) || []).includes(c) ? pi : pj;
        off = P(c).sub(ref);
        off.sub(axis.clone().multiplyScalar(off.dot(axis) / axis.lengthSq()));
      }
      if (!off || off.lengthSq() < 1e-6) off = new THREE.Vector3(1, 0, 0).cross(axis);
      if (off.lengthSq() < 1e-6) off = new THREE.Vector3(0, 1, 0).cross(axis);
      off.normalize().multiplyScalar(0.27);
      const a = pi.clone().lerp(pj, 0.15).add(off), bb = pj.clone().lerp(pi, 0.15).add(off);
      const m2 = a.clone().add(bb).multiplyScalar(0.5);
      cyl.push([a, m2, r * 0.6, ci, i], [m2, bb, r * 0.6, cj, j]);
      if (order === 3) {
        const a3 = pi.clone().lerp(pj, 0.15).sub(off), b3 = pj.clone().lerp(pi, 0.15).sub(off);
        const m3 = a3.clone().add(b3).multiplyScalar(0.5);
        cyl.push([a3, m3, r * 0.6, ci, i], [m3, b3, r * 0.6, cj, j]);
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
    keepInstances(im, cyl.map((c) => c[4]));
    out.push(im);
  }
  const loneSet = new Set(lone);
  const balls = [];
  if (d.style === "lines") balls.push(...lone);
  else for (let i = 0; i < d.atoms.length; i++) if (!(mol.hideH && isH(i))) balls.push(i);
  if (balls.length) {
    const im = new THREE.InstancedMesh(sphereGeo, solidMat, balls.length);
    balls.forEach((i, n) => {
      let r = isH(i) ? H_BALL_R : BALL_R;
      if (loneSet.has(i)) r = isWater(i) ? WATER_R : (isH(i) ? H_BALL_R : ION_R);
      _m.compose(P(i), _q.identity(), _s.set(r, r, r));
      im.setMatrixAt(n, _m);
      im.setColorAt(n, _c.setHex(elColor(d.element[i], k)));
    });
    im.frustumCulled = false;
    keepInstances(im, balls);
    out.push(im);
  }
  return out;
}
// the atom behind each instance and the instance's own matrix, so applyHidden can hide and restore it
function keepInstances(im, atoms) {
  im.userData.atom = Int32Array.from(atoms);
  im.userData.m0 = im.instanceMatrix.array.slice();
}

function atomLabel(k, i) {
  const a = S.mols[k].data.atoms[i];
  return `${a[3]}${a[1]}${a[2]} ${a[4]}${a[5] ? ":" + a[5] : ""}${a[9] ? " [" + a[9] + "]" : ""}`;
}
function atomInfo(k, i) {
  const a = S.mols[k].data.atoms[i], x = S.mols[k].xyz;
  const p = [x[3*i], x[3*i+1], x[3*i+2]];
  const vals = S.maps.filter((m) => m.visible).map((m) => {
    const v = valueAt(m, p);
    return Number.isFinite(v) ? `${m.info.name} ${fmtSigned(v)}${unit(m)}` : null;
  }).filter(Boolean);
  const mol = S.mols.length > 1 ? `  [${S.mols[k].data.name}]` : "";
  return `${a[0]}/${atomLabel(k, i)}  occ ${a[6].toFixed(2)}  B ${a[7].toFixed(1)}${mol}` + (vals.length ? "\n" + vals.join(" · ") : "");
}
const fmtSigned = (v) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(1);

// ------------------------------------------------------------ picking, labels
function pick(clientX, clientY, maxPx) {
  updateCamera(); scene.updateMatrixWorld();
  const rect = renderer.domElement.getBoundingClientRect();
  const v = new THREE.Vector3();
  let best = null, bd = maxPx * maxPx;
  S.mols.forEach((mol, k) => {
    if (!mol.group.visible) return;
    const el = mol.data.element;
    for (let i = 0; i < mol.data.atoms.length; i++) {
      if ((mol.hideH && el[i] === "H") || mol.hidden[i]) continue;
      v.set(mol.xyz[3*i], mol.xyz[3*i+1], mol.xyz[3*i+2]).applyMatrix4(world.matrixWorld).project(camera);
      if (v.z < -1 || v.z > 1) continue;
      const sx = (v.x + 1) / 2 * rect.width, sy = (1 - v.y) / 2 * rect.height;
      const dd = (sx - (clientX - rect.left)) ** 2 + (sy - (clientY - rect.top)) ** 2;
      if (dd < bd || (best && Math.abs(dd - bd) < 4 && v.z < best.z)) { bd = dd; best = { mol: k, idx: i, z: v.z }; }
    }
  });
  return best;
}
function toggleLabel(k, i) {
  const key = `${k}:${i}`;
  if (S.labels.has(key)) { S.labels.get(key).remove(); S.labels.delete(key); }
  else {
    const div = document.createElement("div");
    div.className = "label"; div.textContent = atomLabel(k, i);
    $("#labels").appendChild(div); S.labels.set(key, div);
  }
  redraw();
}
function clearLabels() {
  for (const div of S.labels.values()) div.remove();
  S.labels.clear();
  for (const d of S.distances) { d.div.remove(); overlay.remove(d.line); d.line.geometry.dispose(); }
  S.distances = [];
  redraw();
}
function addDistance(a, b) {
  const A = S.mols[a[0]].xyz, B = S.mols[b[0]].xyz;
  const pa = new THREE.Vector3(A[3*a[1]], A[3*a[1]+1], A[3*a[1]+2]);
  const pb = new THREE.Vector3(B[3*b[1]], B[3*b[1]+1], B[3*b[1]+2]);
  const g = new THREE.BufferGeometry().setFromPoints([pa, pb]);
  const line = new THREE.LineSegments(g, new THREE.LineDashedMaterial({ color: 0x99ffff, dashSize: 0.2, gapSize: 0.15 }));
  line.computeLineDistances();
  overlay.add(line);
  const div = document.createElement("div");
  div.className = "label dist"; div.textContent = `${pa.distanceTo(pb).toFixed(2)} Å`;
  $("#labels").appendChild(div);
  S.distances.push({ a, b, pa, pb, line, div });
}
function updateLabels() {
  const rect = renderer.domElement.getBoundingClientRect();
  const v = new THREE.Vector3();
  scene.updateMatrixWorld();
  for (const [key, div] of S.labels) {
    const [k, i] = key.split(":").map(Number);
    const x = S.mols[k].xyz;
    v.set(x[3*i], x[3*i+1], x[3*i+2]).applyMatrix4(world.matrixWorld).project(camera);
    div.style.display = v.z > -1 && v.z < 1 && S.mols[k].group.visible && !S.mols[k].hidden[i] ? "" : "none";
    div.style.left = `${(v.x + 1) / 2 * rect.width}px`;
    div.style.top = `${(1 - v.y) / 2 * rect.height}px`;
  }
  for (const d of S.distances) {
    const pa = d.pa.clone().applyMatrix4(world.matrixWorld).project(camera);
    const pb = d.pb.clone().applyMatrix4(world.matrixWorld).project(camera);
    d.div.style.left = `${((pa.x + pb.x) / 2 + 1) / 2 * rect.width}px`;
    d.div.style.top = `${(1 - (pa.y + pb.y) / 2) / 2 * rect.height + 11}px`;
  }
}

// --------------------------------------------------------------------- maps
async function decodeMap(info) {
  const bin = atob(info.data);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  const u = new Uint8Array(await new Response(stream).arrayBuffer());   // int16: all high bytes, then all low bytes
  const N = u.length / 2;
  const f = new Float32Array(N);
  for (let i = 0; i < N; i++) f[i] = (((u[i] << 8) | u[N + i]) << 16 >> 16) * info.scale;
  return f;
}

const unit = (m) => (m.info.units === "sigma" ? "σ" : (m.info.unit_label ? " " + m.info.unit_label : ""));
// map value in the units the level is given in (σ of the whole map, or the map's own units)
const toLevelUnits = (m, v) => (m.info.units === "sigma" ? (v - m.info.mean) / m.info.rms : v);

// Stored grids: position = origin + G (i, j, k). "native" maps are the map's own grid over the unit cell
// (periodic, wrap); "block" maps are a piece of that grid around the models; "box" maps are resampled on an
// orthogonal box.
const mat3 = (M, v) => [M[0]*v[0] + M[1]*v[1] + M[2]*v[2], M[3]*v[0] + M[4]*v[1] + M[5]*v[2], M[6]*v[0] + M[7]*v[1] + M[8]*v[2]];
const wrapIndex = (g, n) => ((g % n) + n) % n;

function valueAt(m, p) {
  const { n, origin: o, Ginv, wrap } = m.info;
  const f = mat3(Ginv, [p[0] - o[0], p[1] - o[1], p[2] - o[2]]);
  if (!wrap && f.some((v, a) => v < 0 || v > n[a] - 1)) return NaN;
  const i = f.map((v, a) => (wrap ? Math.floor(v) : Math.min(n[a] - 2, Math.floor(v))));
  const t = f.map((v, a) => v - i[a]);
  const g = (a, d) => (wrap ? wrapIndex(i[a] + d, n[a]) : i[a] + d);
  const D = m.values, syz = n[1] * n[2];
  let s = 0;
  for (let c = 0; c < 8; c++) {
    const dx = c & 1, dy = (c >> 1) & 1, dz = (c >> 2) & 1;
    const w = (dx ? t[0] : 1 - t[0]) * (dy ? t[1] : 1 - t[1]) * (dz ? t[2] : 1 - t[2]);
    s += w * D[g(0, dx) * syz + g(1, dy) * n[2] + g(2, dz)];
  }
  return toLevelUnits(m, s);
}

// Marching cubes (three.js tables): vertices where the contour crosses grid edges, shared between neighbouring
// cells. The page draws the triangle edges (Coot's chicken wire).
const MC_CORNER = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]];
const MC_EDGE = [[0, 0, 0, 0], [1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 1, 1],
  [0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 2], [1, 0, 0, 2], [1, 1, 0, 2], [0, 1, 0, 2]];   // start point, axis
const MCX = MC_CORNER.map((q) => q[0]), MCY = MC_CORNER.map((q) => q[1]), MCZ = MC_CORNER.map((q) => q[2]);

function marchingCubes(m, c, r, level, sign, wire) {
  const { n, origin: o, G, Ginv, wrap } = m.info;
  const D = m.values;
  const off = m.info.units === "sigma" ? m.info.mean : 0, scale = m.info.units === "sigma" ? m.info.rms : 1;
  const L = level * scale;
  // grid cells covering the cube c +/- r (a parallelepiped on a non-orthogonal grid); triangles leaving the cube
  // are dropped, so the contour fills a box around the centre as in Coot
  const ci = mat3(Ginv, [c[0] - o[0], c[1] - o[1], c[2] - o[2]]);
  const lo = [], hi = [];
  for (let a = 0; a < 3; a++) {
    const ext = r * (Math.abs(Ginv[3*a]) + Math.abs(Ginv[3*a+1]) + Math.abs(Ginv[3*a+2]));
    let l = Math.floor(ci[a] - ext), h = Math.ceil(ci[a] + ext);
    if (!wrap) { l = Math.max(0, l); h = Math.min(n[a] - 1, h); }
    lo.push(l); hi.push(h);
  }
  const sx = hi[0] - lo[0], sy = hi[1] - lo[1], sz = hi[2] - lo[2];
  const empty = { verts: new Float32Array(0), index: new Uint32Array(0), segments: new Float32Array(0) };
  if (sx <= 0 || sy <= 0 || sz <= 0) return empty;
  // offset in the value array of each grid plane along each axis (wrapped on the periodic cell grid)
  const stride = [n[1] * n[2], n[2], 1];
  const [X, Y, Z] = [0, 1, 2].map((a) => {
    const t = new Int32Array(hi[a] - lo[a] + 1);
    for (let q = 0; q < t.length; q++) t[q] = (wrap ? wrapIndex(lo[a] + q, n[a]) : lo[a] + q) * stride[a];
    return t;
  });
  const py = sy + 1, pz = sz + 1;
  const edgeVert = new Int32Array((sx + 1) * py * pz * 3).fill(-1);
  const verts = [], inBox = [], tris = [];
  const val = (a, b, d) => sign * (D[X[a] + Y[b] + Z[d]] - off);
  function edgeVertex(a, b, d, axis) {
    const key = ((a * py + b) * pz + d) * 3 + axis;
    let v = edgeVert[key];
    if (v >= 0) return v;
    const v0 = val(a, b, d);
    const v1 = axis === 0 ? val(a + 1, b, d) : axis === 1 ? val(a, b + 1, d) : val(a, b, d + 1);
    const t = (L - v0) / (v1 - v0);
    const fi = lo[0] + a + (axis === 0 ? t : 0), fj = lo[1] + b + (axis === 1 ? t : 0), fk = lo[2] + d + (axis === 2 ? t : 0);
    const x = o[0] + G[0]*fi + G[1]*fj + G[2]*fk, y = o[1] + G[3]*fi + G[4]*fj + G[5]*fk, z = o[2] + G[6]*fi + G[7]*fj + G[8]*fk;
    v = verts.length / 3;
    verts.push(x, y, z);
    inBox.push(Math.abs(x - c[0]) <= r && Math.abs(y - c[1]) <= r && Math.abs(z - c[2]) <= r);
    edgeVert[key] = v;
    return v;
  }
  const ev = new Int32Array(12);
  for (let a = 0; a < sx; a++) for (let b = 0; b < sy; b++) for (let d = 0; d < sz; d++) {
    let cube = 0;
    for (let q = 0; q < 8; q++) if (val(a + MCX[q], b + MCY[q], d + MCZ[q]) < L) cube |= 1 << q;
    const bits = edgeTable[cube];
    if (!bits) continue;
    for (let e = 0; e < 12; e++) {
      if (bits & (1 << e)) { const E = MC_EDGE[e]; ev[e] = edgeVertex(a + E[0], b + E[1], d + E[2], E[3]); }
    }
    for (let t = cube * 16; triTable[t] !== -1; t += 3) {
      const i = ev[triTable[t]], j = ev[triTable[t + 1]], k = ev[triTable[t + 2]];
      if (inBox[i] && inBox[j] && inBox[k]) tris.push(i, j, k);
    }
  }
  const out = { verts: new Float32Array(verts), index: new Uint32Array(tris), segments: new Float32Array(0) };
  if (wire) {   // each triangle edge once
    const seen = new Set(), seg = [];
    const add = (i, j) => {
      const key = i < j ? i * 67108864 + j : j * 67108864 + i;
      if (seen.has(key)) return;
      seen.add(key);
      seg.push(verts[3*i], verts[3*i+1], verts[3*i+2], verts[3*j], verts[3*j+1], verts[3*j+2]);
    };
    for (let t = 0; t < tris.length; t += 3) { add(tris[t], tris[t+1]); add(tris[t+1], tris[t+2]); add(tris[t+2], tris[t]); }
    out.segments = new Float32Array(seg);
  }
  return out;
}

const mapColor = (m, sign) => (m.info.kind === "diff" ? (sign > 0 ? CONTOUR.pos : CONTOUR.neg) : (m.info.color || CONTOUR.map));
function clearMapObj(m) {
  for (const ch of [...m.obj.children]) { m.obj.remove(ch); ch.geometry.dispose(); }
}
// the point the maps are contoured around: the rotation centre moved to the depth of the visible slab
const _view = new THREE.Matrix4();
function viewAxis() {           // model-space direction pointing at the viewer
  const e = _view.makeRotationFromQuaternion(rot.quaternion).elements;
  return new THREE.Vector3(e[2], e[6], e[10]);
}
function contourCenter() { return S.center.clone().addScaledVector(viewAxis(), S.depth); }
function contourMap(m) {
  const cc = contourCenter();
  const c = cc.toArray();
  m.raw = [];
  for (const sign of m.info.kind === "diff" ? [1, -1] : [1]) {
    m.raw.push({ sign, mc: marchingCubes(m, c, S.radius, m.level, sign, true), own: null });
  }
  m.drawn = { center: cc, level: m.level, radius: S.radius };
  drawMap(m);
}
// the contour, less the density of residues hidden by Hide in front
function drawMap(m) {
  clearMapObj(m);
  m.stale = false;
  const hide = CLEAR.hidden;
  for (const r of m.raw || []) {
    const k = r.sign > 0 ? 0 : 1, mc = r.mc;
    if (hide.size && !r.own) r.own = contourOwners(r);
    let seg = mc.segments;
    if (hide.size) {
      const out = new Float32Array(seg.length);
      let w = 0;
      for (let s = 0; s < r.own.length; s++) if (!hide.has(r.own[s])) { out.set(seg.subarray(6*s, 6*s + 6), w); w += 6; }
      seg = out.slice(0, w);
    }
    if (!seg.length) continue;
    const g = new LineSegmentsGeometry();
    g.setPositions(seg);
    m.obj.add(new LineSegments2(g, m.mats[k] || (m.mats[k] = lineMat(0.6, { color: mapColor(m, r.sign) }))));
  }
  redraw();
}
// the residue each contour line belongs to, by its middle
function contourOwners(r) {
  const s = r.mc.segments, own = new Int32Array(s.length / 6);
  for (let n = 0; n < own.length; n++) own[n] = ownerAt((s[6*n] + s[6*n+3]) / 2, (s[6*n+1] + s[6*n+4]) / 2, (s[6*n+2] + s[6*n+5]) / 2);
  return own;
}
let contourTimer = null;
function scheduleContour(now = false) {
  clearTimeout(contourTimer);
  contourTimer = setTimeout(() => {
    const slicing = S.style === "slice";     // slice mode draws slices only, never contours
    for (const m of S.maps) {
      m.obj.visible = m.visible && !slicing;
      if (!m.visible || slicing) continue;
      const d = m.drawn;
      if (d && d.center.distanceTo(contourCenter()) < 0.5 && d.level === m.level && d.radius === S.radius) {
        if (m.stale) drawMap(m);
        continue;
      }
      contourMap(m);
    }
    updateSlice();
    redraw();
  }, now ? 0 : 120);
}

// -------------------------------------------------------------------- slice
// Map section as in Olex2: a plane through the focus that always faces the viewer (so it turns with the
// molecule), at an adjustable depth. It shows the active map (the first visible one): green +, red - (blue for
// 2mFo-DFc-like maps), stronger towards 3 x level, with contour lines at 1, 2, 4, 8 ... x level. The plane is dark
// glass: atoms behind it are dimmed, atoms in front stay bright.
const SLICE = { mesh: null, halo: null, map: null, failed: false, depth: 0, tiltX: 0, tiltY: 0, thick: 0, lock: null,
  off: false };   // off: live plane hidden (saved planes stay)   // lock: {c, Rt}
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

function mapTexture(m) {
  if (m.tex) return m.tex;
  const [n0, n1, n2] = m.info.n;
  const half = new Uint16Array(m.values.length);
  for (let q = 0; q < half.length; q++) half[q] = THREE.DataUtils.toHalfFloat(toLevelUnits(m, m.values[q]));
  const tex = new THREE.Data3DTexture(half, n2, n1, n0);   // x = third grid axis (fastest), z = first
  tex.format = THREE.RedFormat;
  tex.type = THREE.HalfFloatType;
  tex.minFilter = tex.magFilter = THREE.LinearFilter;
  tex.wrapS = tex.wrapT = tex.wrapR = m.info.wrap ? THREE.RepeatWrapping : THREE.ClampToEdgeWrapping;
  tex.unpackAlignment = 1;
  tex.needsUpdate = true;
  return (m.tex = tex);
}
// a slice plane: unit square with the map shader and an outline; its matrix (model coordinates) places it
function makePlaneMesh(outlineColor) {
  const mat = new THREE.ShaderMaterial({
    vertexShader: SLICE_VS, fragmentShader: SLICE_FS, side: THREE.DoubleSide, transparent: true,
    premultipliedAlpha: true, depthWrite: false,
    uniforms: {
      uMap: { value: null }, uPlane: { value: new THREE.Matrix4() }, uGinv: { value: new THREE.Matrix3() },
      uOrigin: { value: new THREE.Vector3() }, uN: { value: new THREE.Vector3() }, uPos: { value: new THREE.Color() },
      uNeg: { value: new THREE.Color() }, uLevel: { value: 1 }, uSigned: { value: 1 }, uWrap: { value: 1 }, uEmpty: { value: 0 },
      uNormal: { value: new THREE.Vector3(0, 0, 1) }, uThick: { value: 0 },
    },
  });
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), mat);
  mesh.matrixAutoUpdate = false;
  mesh.frustumCulled = false;
  mesh.renderOrder = 5;
  const frame = new LineSegmentsGeometry();
  frame.setPositions([-1, -1, 0, 1, -1, 0, 1, -1, 0, 1, 1, 0, 1, 1, 0, -1, 1, 0, -1, 1, 0, -1, -1, 0]);
  const outline = new LineSegments2(frame, lineMat(2, { color: outlineColor }));
  outline.frustumCulled = false;
  mesh.add(outline);
  world.add(mesh);
  return mesh;
}
function setPlaneMap(u, m) {
  u.uMap.value = mapTexture(m);
  u.uGinv.value.set(...m.info.Ginv);
  u.uOrigin.value.set(...m.info.origin);
  u.uN.value.set(...m.info.n);
  u.uWrap.value = m.info.wrap ? 1 : 0;
  u.uPos.value.set(m.info.kind === "diff" ? CONTOUR.pos : (m.info.color || CONTOUR.map));
  u.uNeg.value.set(CONTOUR.neg);
  u.uSigned.value = m.info.kind === "diff" ? 1 : 0;
  u.uLevel.value = m.level;
}
function updateSlice() {
  if (S.style !== "slice" || SLICE.off) {
    if (SLICE.mesh) SLICE.mesh.visible = false;
    if (SLICE.halo) SLICE.halo.visible = false;
    return;
  }
  if (!SLICE.mesh) {
    SLICE.mesh = makePlaneMesh(0xe6c229);
    // atoms the plane cuts through
    const total = S.mols.reduce((t, x) => t + x.data.atoms.length, 0);
    SLICE.halo = new THREE.InstancedMesh(sphereGeo, new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true,
      opacity: 0.45, depthWrite: false }), Math.max(1, total));
    SLICE.halo.count = 0;
    SLICE.halo.frustumCulled = false;
    world.add(SLICE.halo);
  }
  const m = activeMap(), u = SLICE.mesh.material.uniforms;
  u.uEmpty.value = m ? 0 : 1;               // with every map off the plane stays, empty
  if (m) {
    if (SLICE.map !== m) { SLICE.map = m; setPlaneMap(u, m); }
    u.uLevel.value = m.level;
  }
  SLICE.mesh.visible = SLICE.halo.visible = true;
}
// where the live plane is: facing the viewer through the focus (or where it was fixed), tilted, at the chosen depth
// along its normal. Fills _Rt (axes as columns), _M (placement of the unit square), _n, _ex, _ey, _p0.
const _M = new THREE.Matrix4(), _T = new THREE.Matrix4(), _Rt = new THREE.Matrix4();
const _n = new THREE.Vector3(), _ex = new THREE.Vector3(), _ey = new THREE.Vector3(), _p0 = new THREE.Vector3();
function livePlacement() {
  const c = SLICE.lock ? SLICE.lock.c : S.center;
  if (SLICE.lock) _Rt.copy(SLICE.lock.Rt);
  else _Rt.makeRotationFromQuaternion(rot.quaternion).transpose();
  _Rt.multiply(_T.makeRotationX(THREE.MathUtils.degToRad(SLICE.tiltX))).multiply(_T.makeRotationY(THREE.MathUtils.degToRad(SLICE.tiltY)));
  const r = S.radius;
  const d = SLICE.depth + (SLICE.lock ? 0 : S.depth);   // a free plane rides in the middle of the visible slab
  _M.makeTranslation(c.x, c.y, c.z).multiply(_Rt).multiply(_T.makeTranslation(0, 0, d))
    .multiply(_T.makeScale(r, r, 1));
  _ex.setFromMatrixColumn(_Rt, 0); _ey.setFromMatrixColumn(_Rt, 1); _n.setFromMatrixColumn(_Rt, 2);
  _p0.copy(c).addScaledVector(_n, d);
  return r;
}
// per frame: place the live plane; atoms inside it get a halo; saved planes show the map in use (the first visible
// one) at its level, in slice mode only, and stay empty when no map is shown
function sliceFrame() {
  const m = activeMap();
  for (const pl of S.planes) {
    pl.mesh.visible = pl.visible && S.style === "slice";
    const u = pl.mesh.material.uniforms;
    u.uEmpty.value = m ? 0 : 1;
    if (!pl.mesh.visible || !m) continue;
    if (pl.shown !== m) { setPlaneMap(u, m); pl.shown = m; }
    u.uLevel.value = m === pl.map && pl.level != null ? pl.level : m.level;
  }
  if (!SLICE.mesh || !SLICE.mesh.visible) return;
  const r = livePlacement();
  SLICE.mesh.matrix.copy(_M);
  SLICE.mesh.matrixWorldNeedsUpdate = true;
  const u = SLICE.mesh.material.uniforms;
  u.uPlane.value.copy(_M);
  u.uNormal.value.copy(_n).normalize();
  u.uThick.value = SLICE.thick;
  const half = Math.max(0.5, SLICE.thick / 2);
  let k = 0;
  for (const mol of S.mols) {
    if (!mol.group.visible) continue;
    const x = mol.xyz, el = mol.data.element;
    for (let i = 0; i < el.length; i++) {
      if (el[i] === "H" || mol.hidden[i]) continue;
      const dx = x[3*i] - _p0.x, dy = x[3*i+1] - _p0.y, dz = x[3*i+2] - _p0.z;
      if (Math.abs(dx * _n.x + dy * _n.y + dz * _n.z) > half) continue;
      if (Math.abs(dx * _ex.x + dy * _ex.y + dz * _ex.z) > r || Math.abs(dx * _ey.x + dy * _ey.y + dz * _ey.z) > r) continue;
      _m.compose(_v.set(x[3*i], x[3*i+1], x[3*i+2]), _q.identity(), _s.set(0.42, 0.42, 0.42));
      SLICE.halo.setMatrixAt(k++, _m);
    }
  }
  SLICE.halo.count = k;
  SLICE.halo.instanceMatrix.needsUpdate = true;
}

// ------------------------------------------------------------ saved planes
// A saved plane stays in the molecule with its own outline colour and is shown in every map style; the live
// plane is free for the next one. Each viewer's planes are kept in this browser between visits.
const PLANE_COLORS = [0x4fd0ff, 0xff6bd6, 0xffa64d, 0xa0ff70, 0xffffff, 0xb59cff];
const PLANE_KEY = "moorlight-share-planes";
let planeCount = 0;
function addSavedPlane(pl) {
  pl.mesh = makePlaneMesh(pl.color);
  const M = new THREE.Matrix4().makeTranslation(pl.p0.x, pl.p0.y, pl.p0.z).multiply(pl.Rt)
    .multiply(new THREE.Matrix4().makeScale(pl.r, pl.r, 1));
  pl.mesh.matrix.copy(M);
  pl.mesh.matrixWorldNeedsUpdate = true;
  const u = pl.mesh.material.uniforms;
  u.uPlane.value.copy(M);
  if (pl.map) setPlaneMap(u, pl.map);     // a page without maps has empty planes only
  pl.shown = pl.map;
  u.uNormal.value.setFromMatrixColumn(pl.Rt, 2).normalize();
  u.uThick.value = pl.thick;
  S.planes.push(pl);
}
function savePlane() {
  if (S.style !== "slice" || !SLICE.mesh || !SLICE.mesh.visible) {
    status(SLICE.off ? "Turn the live plane on to place a new one." : "Switch to Slice and place the plane first.");
    return;
  }
  const r = livePlacement();
  planeCount += 1;
  addSavedPlane({ name: `Plane ${planeCount}`, color: PLANE_COLORS[(planeCount - 1) % PLANE_COLORS.length],
    p0: _p0.clone(), Rt: _Rt.clone(), r, thick: SLICE.thick, map: activeMap() || SLICE.map || S.maps[0], level: null, visible: true });
  renderPlaneList(); storePlanes(); redraw();
  status(`Plane ${planeCount} saved. Move the live plane to place the next one.`);
}
// planes the page was made with (spec "planes"); they come back on every visit
function loadPresetPlanes() {
  for (const p of DATA.planes || []) {
    const map = S.maps.find((m) => m.info.name === p.map) || S.maps.find((m) => m.info.visible) || S.maps[0];
    if (!map) return;
    const Rt = new THREE.Matrix4().makeBasis(new THREE.Vector3(...p.ex), new THREE.Vector3(...p.ey), new THREE.Vector3(...p.n));
    addSavedPlane({ name: p.name, color: p.color ?? PLANE_COLORS[S.planes.length % PLANE_COLORS.length],
      p0: new THREE.Vector3(...p.p0), Rt, r: p.r, thick: p.thick, map, level: p.level ?? null,
      visible: p.visible !== false, preset: true });
  }
}
// turn the view onto a saved plane: centred on it, facing it
function goToPlane(i) {
  const pl = S.planes[i], m = activeMap();
  pl.visible = true;
  // a preset made on the map in use brings its level along, so the plane, contours and slider agree
  if (pl.level != null && m === pl.map) setLevel(m, pl.level);
  renderPlaneList();
  S.depth = 0;
  animateTo(pl.p0.clone(), new THREE.Quaternion().setFromRotationMatrix(pl.Rt.clone().transpose()));
  status(`${pl.name}${m ? `: ${m.info.name} at ${fmtLevel(m)}` : ": no map shown"}${S.style === "slice" ? "" : " (planes show in Slice)"}`);
}
// a parallel copy of a plane, moved along its normal by the copy offset; it belongs to the viewer
function copyPlane(i) {
  const pl = S.planes[i];
  const off = +$("#copy-offset").value || 0;
  const n = new THREE.Vector3().setFromMatrixColumn(pl.Rt, 2).normalize();
  planeCount += 1;
  addSavedPlane({ name: `${pl.name} ${off >= 0 ? "+" : "\u2212"}${Math.abs(off).toFixed(1)} Å`,
    color: PLANE_COLORS[(planeCount - 1) % PLANE_COLORS.length], p0: pl.p0.clone().addScaledVector(n, off),
    Rt: pl.Rt.clone(), r: pl.r, thick: pl.thick, map: pl.map, level: pl.level, visible: true });
  renderPlaneList(); storePlanes(); redraw();
  status(`Copied ${pl.name}, ${off.toFixed(1)} Å along its normal.`);
}
function removePlane(i) {
  const [pl] = S.planes.splice(i, 1);
  world.remove(pl.mesh);
  pl.mesh.geometry.dispose();
  renderPlaneList(); storePlanes(); redraw();
}
function editPlane(i) {        // the saved plane becomes the live one again
  const pl = S.planes[i];
  removePlane(i);
  if (S.style !== "slice") setStyle("slice");
  if (SLICE.off) setLivePlane(true);
  SLICE.lock = { c: pl.p0.clone(), Rt: pl.Rt.clone() };
  for (const b of [$("#slice-lock"), $("#mini-lock")]) { b.setAttribute("aria-pressed", "true"); b.textContent = "Release"; }
  setTilt("tiltX", 0); setTilt("tiltY", 0); setSliceDepth(0); setThickness(pl.thick);
  status(`${pl.name} is the live plane again.`);
}
function renderPlaneList() {
  const ul = $("#plane-list");
  ul.innerHTML = "";
  S.planes.forEach((pl, i) => {
    const li = document.createElement("li");
    li.className = "plane";
    const chip = "#" + pl.color.toString(16).padStart(6, "0");
    li.innerHTML = `<input type="checkbox" id="plane-vis-${i}"><label for="plane-vis-${i}"><span class="chip" ` +
      `style="border-color:${chip}" aria-hidden="true"></span></label>` +
      `<button type="button" class="go"></button>` +
      `<button type="button">Copy</button><button type="button">Edit</button>` +
      `<button type="button" aria-label="Delete">&times;</button>`;
    const go = li.querySelector(".go");
    go.textContent = pl.name + (pl.preset ? "" : " (yours)");
    go.title = `Look at ${pl.name}${pl.thick ? ` (${pl.thick.toFixed(1)} Å thick)` : ""}`;
    go.addEventListener("click", () => goToPlane(i));
    const cb = li.querySelector("input");
    cb.checked = pl.visible;
    cb.addEventListener("change", () => { pl.visible = cb.checked; storePlanes(); redraw(); });
    const [, copy, edit, del] = li.querySelectorAll("button");
    copy.setAttribute("aria-label", `Copy ${pl.name} at the copy offset`);
    copy.addEventListener("click", () => copyPlane(i));
    edit.addEventListener("click", () => editPlane(i));
    del.setAttribute("aria-label", `Delete ${pl.name}`);
    del.addEventListener("click", () => removePlane(i));
    ul.appendChild(li);
  });
  $("#plane-list").hidden = !S.planes.length;
  $("#planes-empty").hidden = S.planes.length > 0;
  $("#copy-row").hidden = !S.planes.length;
}
function storePlanes() {
  try {
    localStorage.setItem(PLANE_KEY, JSON.stringify({ count: planeCount, planes: S.planes.filter((pl) => !pl.preset).map((pl) => ({ name: pl.name,
      color: pl.color, p0: pl.p0.toArray(), Rt: pl.Rt.toArray(), r: pl.r, thick: pl.thick, map: pl.map ? pl.map.info.name : null,
      level: pl.level, visible: pl.visible })) }));
  } catch (err) { /* storage blocked */ }
}
function loadPlanes() {
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(PLANE_KEY) || "null"); } catch (err) { saved = null; }
  if (!saved || !Array.isArray(saved.planes)) return;
  planeCount = saved.count || 0;
  for (const p of saved.planes) {
    const map = S.maps.find((m) => m.info.name === p.map);
    if (!map) continue;
    addSavedPlane({ name: p.name, color: p.color, p0: new THREE.Vector3(...p.p0), Rt: new THREE.Matrix4().fromArray(p.Rt),
      r: p.r, thick: p.thick, map, level: p.level ?? null, visible: p.visible !== false });
  }
}
function setTilt(axis, deg) {
  SLICE[axis] = Math.max(-90, Math.min(90, deg));
  const id = axis === "tiltX" ? "slice-tilt-x" : "slice-tilt-y";
  $(`#${id}`).value = String(SLICE[axis]);
  $(`#${id}-out`).textContent = `${Math.round(SLICE[axis])}°`;
  redraw();
}
function setLivePlane(on) {
  SLICE.off = !on;
  for (const b of [$("#slice-live"), $("#mini-live")]) {
    b.setAttribute("aria-pressed", String(on));
    b.textContent = on ? "Live plane on" : "Live plane off";
  }
  $("#app").classList.toggle("live-off", !on);
  updateSlice();
  redraw();
}
function setThickness(t) {
  SLICE.thick = Math.max(0, Math.min(4, t));
  $("#slice-thick").value = String(SLICE.thick);
  $("#slice-thick-out").textContent = SLICE.thick ? `${SLICE.thick.toFixed(1)} Å` : "thin";
  redraw();
}
function faceMe() {
  if (SLICE.lock) setSliceLock(false);
  setTilt("tiltX", 0); setTilt("tiltY", 0); setSliceDepth(0);
}
// best plane through the labelled atoms (3 or more): the plane is fixed there and the view turns to face it
function planeThroughLabels() {
  const pts = [];
  for (const key of S.labels.keys()) {
    const [k, i] = key.split(":").map(Number);
    const x = S.mols[k].xyz;
    pts.push(new THREE.Vector3(x[3*i], x[3*i+1], x[3*i+2]));
  }
  if (pts.length < 3) { status("Label three or more atoms first (tap them), then put the plane through them."); return; }
  const c = pts.reduce((a, p) => a.add(p), new THREE.Vector3()).multiplyScalar(1 / pts.length);
  const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (const p of pts) {
    const d = [p.x - c.x, p.y - c.y, p.z - c.z];
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) C[a][b] += d[a] * d[b];
  }
  const { vectors } = eigenSym3(C);          // columns, ascending eigenvalues
  const n = new THREE.Vector3(...vectors.map((r) => r[0])).normalize();
  const ex = new THREE.Vector3(...vectors.map((r) => r[2])).normalize();
  const ey = new THREE.Vector3().crossVectors(n, ex).normalize();
  const Rt = new THREE.Matrix4().makeBasis(ex, ey, n);
  SLICE.lock = { c, Rt };
  for (const b of [$("#slice-lock"), $("#mini-lock")]) { b.setAttribute("aria-pressed", "true"); b.textContent = "Release"; }
  setTilt("tiltX", 0); setTilt("tiltY", 0); setSliceDepth(0);
  if (S.style !== "slice") setStyle("slice");
  animateTo(c, new THREE.Quaternion().setFromRotationMatrix(Rt.clone().transpose()));
  status(`Plane through ${pts.length} labelled atoms (rms out of plane ${Math.sqrt(Math.max(0, eigenSym3(C).values[0]) / pts.length).toFixed(2)} Å).`);
}
// eigen decomposition of a symmetric 3x3 matrix (Jacobi); returns values ascending and vectors as columns
function eigenSym3(A) {
  const a = A.map((r) => r.slice()), v = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let sweep = 0; sweep < 50; sweep++) {
    let off = 0;
    for (let p = 0; p < 2; p++) for (let q = p + 1; q < 3; q++) off += a[p][q] * a[p][q];
    if (off < 1e-20) break;
    for (let p = 0; p < 2; p++) for (let q = p + 1; q < 3; q++) {
      if (Math.abs(a[p][q]) < 1e-30) continue;
      const th = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
      const cs = 1 / Math.sqrt(t * t + 1), sn = t * cs;
      for (let k = 0; k < 3; k++) {
        const akp = a[k][p], akq = a[k][q];
        a[k][p] = cs * akp - sn * akq; a[k][q] = sn * akp + cs * akq;
      }
      for (let k = 0; k < 3; k++) {
        const apk = a[p][k], aqk = a[q][k];
        a[p][k] = cs * apk - sn * aqk; a[q][k] = sn * apk + cs * aqk;
      }
      for (let k = 0; k < 3; k++) {
        const vkp = v[k][p], vkq = v[k][q];
        v[k][p] = cs * vkp - sn * vkq; v[k][q] = sn * vkp + cs * vkq;
      }
    }
  }
  const order = [0, 1, 2].sort((i, j) => a[i][i] - a[j][j]);
  return { values: order.map((i) => a[i][i]), vectors: v.map((r) => order.map((i) => r[i])) };
}
function setSliceLock(on) {
  SLICE.lock = on ? { c: S.center.clone(), Rt: new THREE.Matrix4().makeRotationFromQuaternion(rot.quaternion).transpose() } : null;
  for (const b of [$("#slice-lock"), $("#mini-lock")]) {
    b.setAttribute("aria-pressed", String(on));
    b.textContent = on ? "Release" : "Fix plane";
  }
  if (on) status("Plane fixed in the molecule: rotate to see it from the side; the depth slider moves it along its normal.");
  redraw();
}
renderer.debug.onShaderError = (gl, program, vs, fs) => {
  SLICE.failed = true;
  console.error("shader error", gl.getProgramInfoLog(program), gl.getShaderInfoLog(fs));
  status("The map slice is not available in this browser; back to mesh.");
  setTimeout(() => setStyle("mesh"), 0);
};
function setSliceDepth(z) {
  SLICE.depth = Math.max(-S.slab, Math.min(S.slab, z));
  for (const el of [$("#slice-depth"), $("#mini-depth-range")]) el.value = String(SLICE.depth);
  const txt = `${SLICE.depth >= 0 ? "+" : "−"}${Math.abs(SLICE.depth).toFixed(1)} Å`;
  $("#slice-depth-out").textContent = txt;
  $("#mini-depth-out").textContent = txt;
  redraw();
}

function setStyle(style) {
  if (style === "slice" && SLICE.failed) style = "mesh";
  S.style = style;
  $("#app").classList.toggle("slicing", style === "slice");
  document.querySelectorAll("#style-row button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.style === style)));
  try { localStorage.setItem("moorlight-share-style", style); } catch (err) { /* storage blocked */ }
  scheduleContour(true);
}

function renderMapList() {
  const ul = $("#map-list");
  ul.innerHTML = "";
  S.maps.forEach((m, k) => {
    const li = document.createElement("li");
    li.className = "map";
    const sw = m.info.kind === "diff"
      ? `<span class="sw" aria-hidden="true"><i style="background:var(--pos)"></i><i style="background:var(--neg)"></i></span>`
      : `<span class="sw" aria-hidden="true"><i style="background:var(--map)"></i></span>`;
    li.innerHTML = `<div class="map-head"><input type="checkbox" id="map-vis-${k}"><label for="map-vis-${k}">${sw}` +
      `<span class="name"></span></label></div>` +
      `<div class="lvl"><button type="button" aria-label="Lower contour level">&minus;</button>` +
      `<input type="range" id="map-lvl-${k}" min="0" max="1000" step="1">` +
      `<button type="button" aria-label="Raise contour level">+</button><output for="map-lvl-${k}"></output></div>` +
      `<div class="scale"></div>`;
    li.querySelector(".name").textContent = m.info.name;
    const cb = li.querySelector("input[type=checkbox]");
    cb.checked = m.visible;
    cb.addEventListener("change", () => { m.visible = cb.checked; updateLevels(); scheduleContour(true); });
    const [down, up] = li.querySelectorAll("button");
    down.addEventListener("click", () => changeLevel(m, -levelStep(m)));
    up.addEventListener("click", () => changeLevel(m, levelStep(m)));
    const range = li.querySelector("input[type=range]");
    range.setAttribute("aria-label", `Contour level of ${m.info.name}`);
    bindSlider(range, () => m);
    m.range = range;
    m.out = li.querySelector("output");
    m.scaleEl = li.querySelector(".scale");
    ul.appendChild(li);
  });
  bindSlider($("#mini-level-range"), activeMap);
  updateLevels();
}
const levelStep = (m) => (m.info.kind === "diff" ? 0.5 : 0.25);
const fmtLevel = (m, v = m.level) => `${m.kind === "diff" ? "±" : ""}${v < 10 ? v.toFixed(1) : v.toFixed(0)}${unit(m)}`;
const activeMap = () => S.maps.find((x) => x.visible) || null;
const fmt3 = (v) => { const a = Math.abs(v); return a && (a < 1e-3 || a >= 1e5) ? v.toExponential(2) : String(+v.toPrecision(3)); };
// the map's rms and the level on the map's own (absolute) scale
function scaleText(m) {
  const { mean, rms } = m.info, diff = m.info.kind === "diff";
  if (m.info.units === "sigma") {
    return `map rms ${fmt3(rms)} · level ${diff ? "±" : ""}${fmt3(diff ? m.level * rms : mean + m.level * rms)} (map units)`;
  }
  return `map rms ${fmt3(rms)}${unit(m)} · level = ${fmt3((m.level - (diff ? 0 : mean)) / rms)} × rms`;
}

// Sliders run on a log scale from lo (1 for difference maps, 0.3 otherwise) to the map's largest value, so the
// usual 3-6 level range gets most of the travel and the strongest peaks are still in reach.
function levelRange(m) {
  const lo = m.info.kind === "diff" ? 1 : 0.3;
  return [lo, Math.max(lo * 4, m.maxAbs || lo * 20)];
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
function bindSlider(range, getMap) {
  range.addEventListener("input", () => {
    const m = getMap();
    if (!m) return;
    setLevel(m, posToLevel(m, +range.value), range);
  });
  range.addEventListener("change", () => scheduleContour(true));
}
function updateLevels(source = null) {
  for (const m of S.maps) {
    if (m.out) m.out.textContent = fmtLevel(m);
    if (m.scaleEl) m.scaleEl.textContent = scaleText(m);
    if (m.range && m.range !== source) m.range.value = String(levelToPos(m, m.level));
  }
  S.maps.forEach((m, k) => { const cb = $(`#map-vis-${k}`); if (cb) cb.checked = m.visible; });
  const a = activeMap(), box = $("#mini-level");
  box.classList.toggle("none", !a);
  if (a) {
    $("#mini-level-name").textContent = a.info.name;
    $("#mini-level-out").textContent = fmtLevel(a);
    const r = $("#mini-level-range");
    r.setAttribute("aria-label", `Contour level of ${a.info.name}`);
    if (r !== source) r.value = String(levelToPos(a, a.level));
  }
}
function setLevel(m, v, source = null) {
  m.level = Math.max(0.1, +v.toFixed(2));
  for (const pl of S.planes) if (pl.map === m) pl.level = null;   // planes follow their map's level once it is changed
  if (SLICE.map === m && SLICE.mesh) SLICE.mesh.material.uniforms.uLevel.value = m.level;
  redraw();
  if (!m.visible) m.visible = true;
  updateLevels(source);
  status(`${m.info.name}: contour at ${fmtLevel(m)}`);
  scheduleContour(!source);    // while a slider moves, recontour after a short pause
}
function changeLevel(m, step) { setLevel(m, m.level + step); }

// ------------------------------------------------------------ view and stops
let anim = null;
function animateTo(center, quat) {
  if (REDUCED) {
    S.center.copy(center);
    if (quat) rot.quaternion.copy(quat);
    redraw(); scheduleContour(true);
    return;
  }
  anim = { c0: S.center.clone(), c1: center.clone(), q0: rot.quaternion.clone(), q1: quat ? quat.clone() : null,
    t0: performance.now(), dur: 450 };
}
function animate() {
  if (!anim) return;
  const t = Math.min(1, (performance.now() - anim.t0) / anim.dur);
  const e = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
  S.center.lerpVectors(anim.c0, anim.c1, e);
  if (anim.q1) rot.quaternion.slerpQuaternions(anim.q0, anim.q1, e);
  needsRender = true;
  if (t >= 1) { anim = null; scheduleContour(); }
}

function fitZoom(radius) {
  const half = TAN_HALF * Math.min(1, aspect);
  return Math.min(400, Math.max(2, radius / half));
}

// the starting view of the page (also "Reset view")
function applyView() {
  const s = DATA.start;
  for (const m of S.maps) {
    const v = s.maps ? s.maps[m.info.name] : undefined;
    if (v === false || v === null) m.visible = false;
    else if (typeof v === "number") { m.visible = true; m.level = v; }
    else { m.visible = v === true ? true : m.info.visible; m.level = m.info.level; }
  }
  updateLevels();
  clearLabels();
  for (const [mk, i] of s.labels || []) if (S.mols[mk - 1]) toggleLabel(mk - 1, i);
  for (const [a, b] of s.distances || []) addDistance([a[0] - 1, a[1]], [b[0] - 1, b[1]]);
  S.radius = s.contour_radius || DATA.contour_radius || 10;
  for (const m of S.maps) m.drawn = null;
  S.zoom = fitZoom(s.radius || 10);
  S.slab = s.slab || s.radius || 10;
  SLICE.depth = 0;
  if (SLICE.lock) setSliceLock(false);
  setTilt("tiltX", 0); setTilt("tiltY", 0);
  syncDepthRange();
  S.depth = 0;
  const q = s.orient ? new THREE.Quaternion(...s.orient) : null;
  if (s.style) setStyle(s.style);        // the author's choice (--slice / --open-plane) wins over the remembered one
  animateTo(new THREE.Vector3(...s.center), q);
  const pi = s.plane ? S.planes.findIndex((pl) => pl.name === s.plane) : -1;
  if (pi >= 0) goToPlane(pi);
  scheduleContour(true);
  if (pi < 0) status(S.planes.length && S.style !== "slice" ? `${S.planes.length} plane${S.planes.length > 1 ? "s" : ""}: choose Slice to see ${S.planes.length > 1 ? "them" : "it"}` : "");
}

// ------------------------------------------------------------------- about
// what the page was made from: input files with SHA-256 checksums, the tool, the build time and the spec
function renderAbout() {
  const a = DATA.about, box = $("#about");
  if (!a || !box) return;
  const el = (tag, text, cls) => { const e = document.createElement(tag); if (text != null) e.textContent = text; if (cls) e.className = cls; return e; };
  box.append(el("p", "Everything shown is stored in this file. It needs no network and opens in any current browser " +
    "with WebGL 2; on a phone, open it in the browser (a file preview may not run it)."));
  box.append(el("p", `Made with Moorlight ${a.moorlight}${a.commit ? ` (${a.commit})` : ""} on ${a.built}. ` +
    `Includes three.js ${a.three} (MIT licence).`));
  box.append(el("p", "Input files (SHA-256):"));
  const ul = el("ul");
  for (const f of a.inputs) {
    const li = el("li");
    const mb = f.bytes >= 1e6 ? `${(f.bytes / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(f.bytes / 1e3))} kB`;
    li.append(el("span", `${f.role === "map" ? "Map" : "Model"} "${f.name}": `), el("span", `${f.file}` +
      `${f.columns ? ` (${f.columns})` : ""}, ${mb}`), el("div", f.sha256, "hash"));
    ul.append(li);
  }
  box.append(ul);
  const det = el("details");
  det.append(el("summary", "Spec used to build this page"), el("pre", JSON.stringify(a.spec, null, 1)));
  box.append(det);
}

// ---------------------------------------------------------------- structure
function renderModelList() {
  const ul = $("#mol-list");
  ul.innerHTML = "";
  S.mols.forEach((mol, k) => {
    const li = document.createElement("li");
    li.className = "mol";
    const chip = "#" + carbonColor(k).toString(16).padStart(6, "0");
    li.innerHTML = `<input type="checkbox" id="mol-vis-${k}"><label for="mol-vis-${k}">` +
      `<span class="chip" style="background:${chip}" aria-hidden="true"></span><span class="name"></span></label>`;
    li.querySelector(".name").textContent = mol.data.name;
    const cb = li.querySelector("input");
    cb.checked = mol.group.visible;
    cb.addEventListener("change", () => setMolVisible(k, cb.checked));
    ul.appendChild(li);
  });
  updateAtomsButton();
}
function setMolVisible(k, on) {
  S.mols[k].group.visible = on;
  const cb = $(`#mol-vis-${k}`);
  if (cb) cb.checked = on;
  updateAtomsButton();
  CLEAR.focus = null;                 // contacts and density owners change with the visible molecules
  for (const m of S.maps) if (m.raw) { m.raw.forEach((r) => { r.own = null; }); m.stale = true; }
  scheduleContour();
  redraw();
}

// ------------------------------------------------------------ hide in front
// Residues in front of the one at the centre that cover it on screen are hidden, with the map density around
// them, so nothing stands between you and the centre. Residues bonded to it or within CONTACT of it stay (a metal
// keeps its ligands). Map density belongs to the residue of the nearest atom within OWN_R.
const CLEAR = { on: true, hidden: new Set(), focus: null, keep: new Set() };
const BLOCK_R = 1.5, POINT_R = 2.0, FOCUS_R = 2.5, CONTACT = 2.6, OWN_R = 2.2;
let NRES = 0;
function indexResidues(mol) {
  const ids = new Map(), a = mol.data.atoms;
  mol.res = new Int32Array(a.length);
  for (let i = 0; i < a.length; i++) {
    const key = `${a[i][0]}|${a[i][1]}|${a[i][2]}|${a[i][9] || ""}`;   // neighbour copies are residues of their own
    if (!ids.has(key)) ids.set(key, NRES++);
    mol.res[i] = ids.get(key);
  }
}
const OWN = new Map();
const ownCell = (v) => Math.floor(v / OWN_R) + 1024;
function indexOwners() {
  S.mols.forEach((mol, k) => {
    const x = mol.xyz, el = mol.data.element;
    for (let i = 0; i < el.length; i++) {
      if (el[i] === "H") continue;
      const key = (ownCell(x[3*i]) * 2048 + ownCell(x[3*i+1])) * 2048 + ownCell(x[3*i+2]);
      (OWN.get(key) || OWN.set(key, []).get(key)).push(k, i);
    }
  });
}
function ownerAt(px, py, pz) {
  const cx = ownCell(px), cy = ownCell(py), cz = ownCell(pz);
  let best = -1, bd = OWN_R * OWN_R;
  for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) for (let c = -1; c <= 1; c++) {
    const list = OWN.get(((cx + a) * 2048 + cy + b) * 2048 + cz + c);
    if (!list) continue;
    for (let t = 0; t < list.length; t += 2) {
      const mol = S.mols[list[t]];
      if (!mol.group.visible) continue;
      const i = list[t + 1], x = mol.xyz;
      const d = (x[3*i] - px) ** 2 + (x[3*i+1] - py) ** 2 + (x[3*i+2] - pz) ** 2;
      if (d < bd) { bd = d; best = mol.res[i]; }
    }
  }
  return best;
}
function contacts(fk, focus, pts) {
  const keep = new Set();
  if (fk < 0) return keep;
  const d = S.mols[fk].data, res = S.mols[fk].res;
  for (let b = 0; b < d.bonds.length; b += 2) {
    const ri = res[d.bonds[b]], rj = res[d.bonds[b + 1]];
    if (ri === focus) keep.add(rj);
    if (rj === focus) keep.add(ri);
  }
  const C2 = CONTACT * CONTACT;
  for (const mol of S.mols) {
    if (!mol.group.visible) continue;
    const x = mol.xyz, el = mol.data.element;
    for (let i = 0; i < el.length; i++) {
      if (el[i] === "H" || keep.has(mol.res[i])) continue;
      for (let p = 0; p < pts.length; p += 3) {
        if ((x[3*i] - pts[p]) ** 2 + (x[3*i+1] - pts[p+1]) ** 2 + (x[3*i+2] - pts[p+2]) ** 2 < C2) { keep.add(mol.res[i]); break; }
      }
    }
  }
  keep.delete(focus);
  return keep;
}
const _E = new THREE.Matrix4();
function updateClear() {
  const hidden = new Set();
  if (CLEAR.on) {
    const f = contourCenter(), c = S.center;
    // the residue at the centre: that of the nearest atom within FOCUS_R; with none, the centre point itself
    let fk = -1, fi = -1, bd = FOCUS_R * FOCUS_R;
    S.mols.forEach((mol, k) => {
      if (!mol.group.visible) return;
      const x = mol.xyz, el = mol.data.element;
      for (let i = 0; i < el.length; i++) {
        if (el[i] === "H") continue;
        const d = (x[3*i] - f.x) ** 2 + (x[3*i+1] - f.y) ** 2 + (x[3*i+2] - f.z) ** 2;
        if (d < bd) { bd = d; fk = k; fi = i; }
      }
    });
    const focus = fk < 0 ? -1 : S.mols[fk].res[fi];
    const pts = [];
    if (fk < 0) pts.push(f.x, f.y, f.z);
    else {
      const mol = S.mols[fk], x = mol.xyz;
      for (let i = 0; i < mol.res.length; i++) if (mol.res[i] === focus && mol.data.element[i] !== "H") pts.push(x[3*i], x[3*i+1], x[3*i+2]);
    }
    if (focus !== CLEAR.focus) { CLEAR.focus = focus; CLEAR.keep = contacts(fk, focus, pts); }
    // in view coordinates (x right, y up, z towards you, Å from the rotation centre)
    const e = _E.makeRotationFromQuaternion(rot.quaternion).elements;
    const fp = [];
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (let p = 0; p < pts.length; p += 3) {
      const dx = pts[p] - c.x, dy = pts[p+1] - c.y, dz = pts[p+2] - c.z;
      const vx = e[0]*dx + e[4]*dy + e[8]*dz, vy = e[1]*dx + e[5]*dy + e[9]*dz;
      fp.push(vx, vy, e[2]*dx + e[6]*dy + e[10]*dz);
      x0 = Math.min(x0, vx); x1 = Math.max(x1, vx); y0 = Math.min(y0, vy); y1 = Math.max(y1, vy);
    }
    const R = fk < 0 ? POINT_R : BLOCK_R, R2 = R * R, front = S.depth + S.slab;
    for (const mol of S.mols) {
      if (!mol.group.visible) continue;
      const x = mol.xyz, el = mol.data.element, res = mol.res;
      for (let i = 0; i < el.length; i++) {
        const r = res[i];
        if (r === focus || el[i] === "H" || hidden.has(r) || CLEAR.keep.has(r)) continue;
        const dx = x[3*i] - c.x, dy = x[3*i+1] - c.y, dz = x[3*i+2] - c.z;
        const vx = e[0]*dx + e[4]*dy + e[8]*dz;
        if (vx < x0 - R || vx > x1 + R) continue;
        const vy = e[1]*dx + e[5]*dy + e[9]*dz;
        if (vy < y0 - R || vy > y1 + R) continue;
        const vz = e[2]*dx + e[6]*dy + e[10]*dz;
        if (vz > front) continue;          // cut off by the slab anyway
        for (let p = 0; p < fp.length; p += 3) {
          if (vz > fp[p+2] + 0.3 && (vx - fp[p]) ** 2 + (vy - fp[p+1]) ** 2 < R2) { hidden.add(r); break; }
        }
      }
    }
  }
  if (hidden.size === CLEAR.hidden.size && [...hidden].every((r) => CLEAR.hidden.has(r))) return;
  CLEAR.hidden = hidden;
  S.mols.forEach((mol, k) => {
    for (let i = 0; i < mol.res.length; i++) mol.hidden[i] = hidden.has(mol.res[i]) ? 1 : 0;
    applyHidden(k);
  });
  for (const m of S.maps) if (m.raw) { if (m.obj.visible) drawMap(m); else m.stale = true; }
}
function setClear(on) {
  CLEAR.on = on;
  $("#front").setAttribute("aria-pressed", String(on));
  try { localStorage.setItem("moorlight-share-front", on ? "1" : "0"); } catch (err) { /* storage blocked */ }
  redraw();
}
function updateAtomsButton() {
  $("#mini-atoms").setAttribute("aria-pressed", String(S.mols.some((m) => m.group.visible)));
}
function toggleStructure() {
  const show = !S.mols.some((m) => m.group.visible);
  S.mols.forEach((m, k) => setMolVisible(k, show));
}

// ------------------------------------------------------------------ input
const el = renderer.domElement;
const pointers = new Map();
let gesture = null, lastTap = null;
el.addEventListener("contextmenu", (e) => e.preventDefault());
el.addEventListener("pointerdown", (e) => {
  e.preventDefault();
  try { el.setPointerCapture(e.pointerId); } catch (err) { /* ignore */ }
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  if (pointers.size === 1) {
    gesture = { t0: performance.now(), x0: e.clientX, y0: e.clientY, moved: false, multi: false,
      button: e.button, pan: e.ctrlKey || e.metaKey || e.shiftKey, touch: e.pointerType !== "mouse" };
    anim = null;
  } else if (gesture) { gesture.multi = true; gesture.moved = true; }
});
el.addEventListener("pointermove", (e) => {
  const p = pointers.get(e.pointerId);
  if (!p || !gesture) return;
  const prev = { x: p.x, y: p.y };
  if (pointers.size >= 2) {
    const pts = [...pointers.entries()];
    const geom = (sub) => {
      const ps = pts.map(([id, q]) => (id === e.pointerId ? sub : q));
      const cx = ps.reduce((s, q) => s + q.x, 0) / ps.length, cy = ps.reduce((s, q) => s + q.y, 0) / ps.length;
      const d = Math.hypot(ps[0].x - ps[1].x, ps[0].y - ps[1].y);
      return { cx, cy, d };
    };
    const before = geom(prev), after = geom({ x: e.clientX, y: e.clientY });
    p.x = e.clientX; p.y = e.clientY;
    if (before.d > 0 && after.d > 0) S.zoom = Math.min(400, Math.max(2, S.zoom * before.d / after.d));
    pan(after.cx - before.cx, after.cy - before.cy);
    redraw();
    return;
  }
  p.x = e.clientX; p.y = e.clientY;
  if (!gesture.moved && Math.hypot(e.clientX - gesture.x0, e.clientY - gesture.y0) < (gesture.touch ? 8 : 3)) return;
  gesture.moved = true;
  const dx = e.clientX - prev.x, dy = e.clientY - prev.y;
  if (gesture.button === 1 || gesture.button === 2 || gesture.pan) {   // right button = two-finger click on a trackpad
    pan(dx, dy);
  } else {
    const axis = new THREE.Vector3(dy, dx, 0);
    const ang = axis.length() * 0.008;
    if (ang > 0) {
      rot.quaternion.premultiply(new THREE.Quaternion().setFromAxisAngle(axis.normalize(), ang));
      redraw();
    }
  }
});
function pan(dx, dy) {
  const h = 2 * S.zoom * TAN_HALF;
  const s = h / el.clientHeight;
  S.center.add(new THREE.Vector3(-dx * s, dy * s, 0).applyQuaternion(rot.quaternion.clone().invert()));
  redraw(); scheduleContour();
}
function endPointer(e) {
  if (!pointers.has(e.pointerId)) return;
  pointers.delete(e.pointerId);
  if (pointers.size || !gesture) return;
  const g = gesture;
  gesture = null;
  if (g.moved || g.multi) scheduleContour();
  if (g.moved || g.multi || e.type === "pointercancel" || performance.now() - g.t0 > 600) return;
  const p = pick(e.clientX, e.clientY, g.touch ? 24 : 12);
  if (!p) return;
  const now = performance.now();
  if (lastTap && lastTap.mol === p.mol && lastTap.idx === p.idx && now - lastTap.t < 400) {
    toggleLabel(p.mol, p.idx);   // undo the first tap's label toggle
    const x = S.mols[p.mol].xyz;
    S.depth = 0;               // the tapped atom becomes the rotation centre, with the slab around it
    animateTo(new THREE.Vector3(x[3*p.idx], x[3*p.idx+1], x[3*p.idx+2]), null);
    lastTap = null;
    return;
  }
  lastTap = { ...p, t: now };
  toggleLabel(p.mol, p.idx);
  status(atomInfo(p.mol, p.idx));
}
el.addEventListener("pointerup", endPointer);
el.addEventListener("pointercancel", endPointer);
let wheelLevel = null;   // unrounded level while scrolling, so small trackpad steps add up
el.addEventListener("wheel", (e) => {   // two-finger scroll (or the wheel) sets the level of the first visible map; pinch arrives as Ctrl+wheel
  e.preventDefault();
  if (e.ctrlKey) { S.zoom = Math.min(400, Math.max(2, S.zoom * Math.exp(e.deltaY * 0.01))); redraw(); return; }
  const m = activeMap();
  if (!m) return;
  const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? el.clientHeight : 1;
  if (!wheelLevel || wheelLevel.m !== m || Math.abs(wheelLevel.v - m.level) > 0.01) wheelLevel = { m, v: m.level };
  wheelLevel.v = Math.min(levelRange(m)[1], Math.max(0.1, wheelLevel.v * Math.exp(-e.deltaY * unit * 0.0015)));
  setLevel(m, wheelLevel.v, "wheel");
}, { passive: false });
window.addEventListener("keydown", (e) => {
  if (e.target.closest && e.target.closest("input, select")) return;
  if (e.key === "f" || e.key === "F") setFull(!$("#app").classList.contains("full"));
  else if ((e.key === "l" || e.key === "L") && S.style === "slice") setSliceLock(!SLICE.lock);
  else if (e.key === "a" || e.key === "A") toggleStructure();
  else if (e.key === "c" || e.key === "C") setClear(!CLEAR.on);
  else if (e.key === "," || e.key === ".") moveDepth(e.key === "." ? 0.5 : -0.5);
  else if (e.key === "[" || e.key === "]") setSlab(S.slab * (e.key === "]" ? 1.2 : 1 / 1.2));
  else if ((e.key === "s" || e.key === "S") && S.style === "slice") savePlane();
});

document.querySelectorAll("#style-row button").forEach((b) => b.addEventListener("click", () => setStyle(b.dataset.style)));
try {
  const st = localStorage.getItem("moorlight-share-style");
  if (st === "slice") S.style = st;
} catch (err) { /* storage blocked */ }
document.querySelectorAll("#style-row button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.style === S.style)));
$("#app").classList.toggle("slicing", S.style === "slice");
for (const el of [$("#slice-depth"), $("#mini-depth-range")]) el.addEventListener("input", () => setSliceDepth(+el.value));
for (const b of [$("#slice-lock"), $("#mini-lock")]) b.addEventListener("click", () => setSliceLock(!SLICE.lock));
$("#slice-tilt-x").addEventListener("input", (e) => setTilt("tiltX", +e.target.value));
$("#slice-tilt-y").addEventListener("input", (e) => setTilt("tiltY", +e.target.value));
$("#slice-face").addEventListener("click", faceMe);
for (const b of [$("#slice-live"), $("#mini-live")]) b.addEventListener("click", () => setLivePlane(SLICE.off));
$("#slice-thick").addEventListener("input", (e) => setThickness(+e.target.value));
$("#copy-offset").addEventListener("input", (e) => {
  const v = +e.target.value;
  $("#copy-offset-out").textContent = `${v >= 0 ? "+" : "\u2212"}${Math.abs(v).toFixed(1)} Å`;
});
for (const b of [$("#slice-save"), $("#mini-save")]) b.addEventListener("click", savePlane);
$("#slice-atoms").addEventListener("click", planeThroughLabels);
$("#mini-atoms").addEventListener("click", toggleStructure);
$("#front").addEventListener("click", () => setClear(!CLEAR.on));
try { if (localStorage.getItem("moorlight-share-front") === "0") setClear(false); } catch (err) { /* storage blocked */ }
$("#reset").addEventListener("click", applyView);
$("#clear").addEventListener("click", () => { clearLabels(); status(""); });
// ---------------------------------------------------------------- depth bar
// A scrollbar for depth on the right of the view: the bar is the structure from front (top) to back (bottom),
// the window on it is the visible slab around the focus. Dragging the window moves the focus into or out of the
// screen; dragging its ends changes the slab; tapping the bar jumps there. The yellow tick is the live slice.
const DB = { front: 10, back: -10, drag: null };
const _zrow = new THREE.Matrix4();
function depthExtent() {
  _zrow.makeRotationFromQuaternion(rot.quaternion);
  const e = _zrow.elements, zx = e[2], zy = e[6], zz = e[10];
  let lo = Infinity, hi = -Infinity;
  const c = S.center;
  for (const mol of S.mols) {
    if (!mol.group.visible) continue;
    const x = mol.xyz, el = mol.data.element;
    for (let i = 0; i < el.length; i++) {
      if (el[i] === "H") continue;
      const z = zx * (x[3*i] - c.x) + zy * (x[3*i+1] - c.y) + zz * (x[3*i+2] - c.z);
      if (z < lo) lo = z;
      if (z > hi) hi = z;
    }
  }
  if (!Number.isFinite(lo)) { lo = -S.slab; hi = S.slab; }
  DB.front = Math.max(hi, S.depth + S.slab) + 1;    // z towards the viewer, relative to the rotation centre
  DB.back = Math.min(lo, S.depth - S.slab) - 1;
}
const depthToFrac = (z) => (DB.front - z) / (DB.front - DB.back);
function updateDepthBar() {
  depthExtent();
  const bar = $("#depthbar"), H = bar.clientHeight;
  const top = depthToFrac(S.depth + S.slab), bot = depthToFrac(S.depth - S.slab);
  const win = $("#depth-window");
  win.style.top = `${top * H}px`;
  win.style.height = `${Math.max(14, (bot - top) * H)}px`;
  const tick = $("#depth-plane");
  const live = S.style === "slice" && !SLICE.off && SLICE.mesh && SLICE.mesh.visible && !SLICE.lock && !SLICE.tiltX && !SLICE.tiltY;
  tick.hidden = !live;
  if (live) tick.style.top = `${depthToFrac(S.depth + SLICE.depth) * H}px`;
  $("#depth-pivot").style.top = `${depthToFrac(0) * H}px`;
  bar.setAttribute("aria-valuetext", `slab ±${S.slab.toFixed(1)} Å, ${S.depth >= 0 ? "+" : "\u2212"}${Math.abs(S.depth).toFixed(1)} Å from the rotation centre`);
}
function moveDepth(dz) {        // dz > 0: the visible slab goes into the screen; the rotation centre stays
  S.depth -= dz;
  redraw(); scheduleContour();
}
function setSlab(v) {
  S.slab = Math.max(0.8, Math.min(200, v));
  syncDepthRange(); redraw();
}
{
  const bar = $("#depthbar");
  const zAt = (clientY) => {
    const r = bar.getBoundingClientRect();
    const f = Math.min(1, Math.max(0, (clientY - r.top) / r.height));
    return DB.front - f * (DB.front - DB.back);
  };
  bar.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    e.stopPropagation();
    try { bar.setPointerCapture(e.pointerId); } catch (err) { /* ignore */ }
    const t = e.target.closest ? e.target : null;
    const grip = t && t.classList && t.classList.contains("grip");
    const inWin = t && t.closest && t.closest("#depth-window");
    if (grip) DB.drag = { mode: "slab" };
    else {
      if (!inWin) moveDepth(S.depth - zAt(e.clientY));      // tap on the bar: jump there
      DB.drag = { mode: "move", y: e.clientY };
    }
    bar.classList.add("active");
  });
  bar.addEventListener("pointermove", (e) => {
    if (!DB.drag) return;
    if (DB.drag.mode === "slab") {
      setSlab(Math.abs(zAt(e.clientY) - S.depth));
    } else {
      const dy = e.clientY - DB.drag.y;
      DB.drag.y = e.clientY;
      moveDepth(dy * (DB.front - DB.back) / bar.clientHeight);
    }
    updateDepthBar();
  });
  const end = () => { DB.drag = null; bar.classList.remove("active"); scheduleContour(true); redraw(); };
  bar.addEventListener("pointerup", end);
  bar.addEventListener("pointercancel", end);
}
function syncDepthRange() {
  for (const el of [$("#slice-depth"), $("#mini-depth-range")]) { el.min = String(-S.slab); el.max = String(S.slab); }
  setSliceDepth(SLICE.depth);
}
$("#hyd").addEventListener("click", (e) => {
  const show = S.mols.some((m) => m.hideH);
  S.mols.forEach((m, k) => { m.hideH = !show; buildMolecule(k); });
  e.currentTarget.setAttribute("aria-pressed", String(show));
});
const svgIcon = (d) => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" ` +
  `stroke-linejoin="round" aria-hidden="true"><path d="${d}"/></svg>`;
const ICON_FULL = svgIcon("M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5");
const ICON_MENU = svgIcon("M4 6h16M4 12h16M4 18h16");
$("#fullscreen").innerHTML = ICON_FULL;
function setFull(on) {
  $("#app").classList.toggle("full", on);
  const b = $("#fullscreen");
  b.setAttribute("aria-pressed", String(on));
  b.setAttribute("aria-label", on ? "Show menus" : "Full screen view");
  b.innerHTML = on ? ICON_MENU : ICON_FULL;
  try { localStorage.setItem("moorlight-share-full", on ? "1" : "0"); } catch (err) { /* storage blocked */ }
}
$("#fullscreen").addEventListener("click", () => setFull(!$("#app").classList.contains("full")));
try { if (localStorage.getItem("moorlight-share-full") === "1") setFull(true); } catch (err) { /* storage blocked */ }

// ------------------------------------------------------------------- start
async function start() {
  DATA.models.forEach((d, k) => {
    const mol = { data: d, group: new THREE.Group(), xyz: Float32Array.from(d.xyz), hideH: !d.hydrogens,
      hidden: new Uint8Array(d.atoms.length) };
    indexResidues(mol);
    world.add(mol.group);
    S.mols.push(mol);
    buildMolecule(k);
  });
  indexOwners();
  $("#hyd").setAttribute("aria-pressed", String(S.mols.some((m) => !m.hideH)));
  resize();
  render();
  if (DATA.maps.length && typeof DecompressionStream === "undefined") {
    $("#loading").textContent = "This browser cannot unpack the maps. Use a current Safari, Chrome or Firefox.";
  } else {
    for (const info of DATA.maps) {
      const m = { info, values: await decodeMap(info), visible: info.visible, level: info.level, kind: info.kind,
        obj: new THREE.Group(), mats: [], drawn: null, vol: null, tex: null };
      let mx = 0;
      for (const v of m.values) { const a = Math.abs(toLevelUnits(m, v)); if (a > mx) mx = a; }
      m.maxAbs = mx;
      world.add(m.obj);
      S.maps.push(m);
    }
    $("#loading").hidden = true;
  }
  renderMapList();
  renderModelList();
  renderAbout();
  loadPresetPlanes();
  loadPlanes();
  renderPlaneList();
  applyView();
}
start().catch((err) => {
  $("#loading").hidden = false;
  $("#loading").textContent = `Could not show the snapshot: ${err.message}`;
  console.error(err);
});
