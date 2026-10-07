// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

import createTinyopt from './tinyopt.mjs';
import { PROBLEMS, cost } from './problems.js';
import { SOLVERS, createOptimizer } from './optimizer.js';

const GRID = 160;        // surface resolution
const WIDTH = 10;        // scene size of the longest domain side
const HEIGHT = 5;        // scene height of the highest cost
const MAX_ITERS = 200;
const SOLVER_PARAM_DEFAULTS = {
  0: { key: 'lm_damping_init', defaultValue: 1e-4, minExp: -8, maxExp: 2, step: 0.05 },
  1: { key: 'check_min_hessian_diagonal', defaultValue: 1e-8, minExp: -12, maxExp: -2, step: 0.05 },
  2: { key: 'gd_learning_rate', defaultValue: 1e-3, minExp: -8, maxExp: 2, step: 0.05 },
  3: { key: 'cg_step_size', defaultValue: 0.25, minExp: -6, maxExp: 1, step: 0.05 },
  4: { key: 'dogleg_radius_init', defaultValue: 1, minExp: -2, maxExp: 4, step: 0.05 },
  5: { key: 'bfgs_step_size', defaultValue: 1, minExp: -4, maxExp: 2, step: 0.05 },
  6: { key: 'lbfgs_step_size', defaultValue: 1, minExp: -4, maxExp: 2, step: 0.05 },
};
const solverParamValues = new Map();

function formatParamValue(value) {
  if (value === 0) return '0';
  if (value >= 1e4 || value < 1e-3) return value.toExponential(2);
  return value.toFixed(3).replace(/\.0+$|0+$/, '');
}

function solverParamConfig(solverId) {
  return SOLVER_PARAM_DEFAULTS[solverId] ?? null;
}

function solverParamOption(solverId) {
  const cfg = solverParamConfig(solverId);
  if (!cfg) return {};
  const value = solverParamValues.get(solverId) ?? cfg.defaultValue;
  return { [cfg.key]: value };
}

const $ = (id) => document.getElementById(id);
const optimizer = await createOptimizer(createTinyopt);
$('status').textContent = '';

$('panel-hide').addEventListener('click', () => {
  $('panel').hidden = true;
  $('panel-show').hidden = false;
  $('panel-show').focus();
});
$('panel-show').addEventListener('click', () => {
  $('panel').hidden = false;
  $('panel-show').hidden = true;
  $('panel-hide').focus();
});

// Scene
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
$('scene').appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x14161a);
const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 200);
camera.position.set(11, 10, 13);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0, 1.5, 0);
scene.add(new THREE.AmbientLight(0xffffff, 0.8));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(6, 12, 8);
scene.add(sun);

function resize() {
  renderer.setSize(window.innerWidth, window.innerHeight);
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();

// State
let problem = PROBLEMS[0];
let view = null;   // {scale, cx, cy, fmax}
let surface = null;
const dynamic = new THREE.Group(); // minima, start marker and trajectories
scene.add(dynamic);
const solverGroups = new Map();    // solver id -> THREE.Group
const hidden = new Set();
let start = null;

const toScene = (x, y, f) => new THREE.Vector3(
  (x - view.cx) * view.scale,
  HEIGHT * Math.log1p(Math.min(f, view.fmax)) / Math.log1p(view.fmax) + 0.06,
  (y - view.cy) * view.scale);
const inDomain = (x, y) => x >= problem.domain.x[0] && x <= problem.domain.x[1] &&
                           y >= problem.domain.y[0] && y <= problem.domain.y[1];
const lift = (x, y) => toScene(x, y, cost(problem, x, y));

function buildSurface() {
  const [x0, x1] = problem.domain.x;
  const [y0, y1] = problem.domain.y;
  view = { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, scale: WIDTH / Math.max(x1 - x0, y1 - y0),
           fmax: 1 };
  const costs = new Float64Array((GRID + 1) * (GRID + 1));
  for (let j = 0, k = 0; j <= GRID; ++j)
    for (let i = 0; i <= GRID; ++i, ++k) {
      costs[k] = cost(problem, x0 + (x1 - x0) * i / GRID, y0 + (y1 - y0) * j / GRID);
      view.fmax = Math.max(view.fmax, costs[k]);
    }
  const positions = new Float32Array(costs.length * 3);
  const colors = new Float32Array(costs.length * 3);
  const color = new THREE.Color();
  for (let j = 0, k = 0; j <= GRID; ++j)
    for (let i = 0; i <= GRID; ++i, ++k) {
      const p = toScene(x0 + (x1 - x0) * i / GRID, y0 + (y1 - y0) * j / GRID, costs[k]);
      p.y -= 0.06;
      positions.set([p.x, p.y, p.z], 3 * k);
      color.setHSL(0.66 * (1 - p.y / HEIGHT), 0.75, 0.45).toArray(colors, 3 * k);
    }
  const indices = [];
  for (let j = 0; j < GRID; ++j)
    for (let i = 0; i < GRID; ++i) {
      const a = j * (GRID + 1) + i;
      indices.push(a, a + GRID + 1, a + 1, a + 1, a + GRID + 1, a + GRID + 2);
    }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  if (surface) { scene.remove(surface); surface.geometry.dispose(); }
  surface = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({
    vertexColors: true, side: THREE.DoubleSide, roughness: 0.8, metalness: 0.0 }));
  scene.add(surface);
}

function marker(position, color, radius) {
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(radius, 16, 12),
                              new THREE.MeshBasicMaterial({ color }));
  mesh.position.copy(position);
  return mesh;
}

function clearDynamic() {
  dynamic.traverse((o) => { o.geometry?.dispose(); o.material?.dispose(); });
  dynamic.clear();
  solverGroups.clear();
}

function drawSolver(solver, index) {
  const previous = solverGroups.get(solver.id);
  if (previous) {
    dynamic.remove(previous);
    previous.traverse((o) => { o.geometry?.dispose(); o.material?.dispose(); });
  }
  const result = optimizer.minimize(problem, start, {
    solverType: solver.id,
    maxIters: MAX_ITERS,
    ...solverParamOption(solver.id),
  });
  const group = new THREE.Group();
  group.visible = !hidden.has(solver.id);
  const points = result.trajectory.filter(([x, y]) => inDomain(x, y)).map(([x, y]) => lift(x, y));
  if (points.length > 1) {
    group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),
                             new THREE.LineBasicMaterial({ color: solver.color })));
  }
  if (points.length > 0) {
    group.add(new THREE.Points(new THREE.BufferGeometry().setFromPoints(points),
                               new THREE.PointsMaterial({ color: solver.color, size: 0.08 })));
    group.add(marker(points.at(-1), solver.color, 0.14));
  }
  dynamic.add(group);
  solverGroups.set(solver.id, group);
  $('legend').children[index].querySelector('.stats').textContent =
      result.ok ? `${result.trajectory.length} pts, f=${result.cost.toExponential(1)}` : 'failed';
}

function solve() {
  clearDynamic();
  for (const [mx, my] of problem.minima) dynamic.add(marker(lift(mx, my), 0xffffff, 0.12));
  dynamic.add(marker(lift(...start), 0xff00ff, 0.2));
  SOLVERS.forEach(drawSolver);
}

function randomStart() {
  const [x0, x1] = problem.domain.x;
  const [y0, y1] = problem.domain.y;
  const margin = 0.1;
  start = [x0 + (x1 - x0) * (margin + (1 - 2 * margin) * Math.random()),
           y0 + (y1 - y0) * (margin + (1 - 2 * margin) * Math.random())];
  solve();
}

function selectProblem(p) {
  problem = p;
  buildSurface();
  randomStart();
}

// UI
for (const p of PROBLEMS) $('problem').add(new Option(p.name, p.id));
$('problem').addEventListener('change', (e) =>
  selectProblem(PROBLEMS.find((p) => p.id === e.target.value)));
$('random').addEventListener('click', randomStart);
for (const solver of SOLVERS) {
  const config = solverParamConfig(solver.id);
  const li = document.createElement('li');
  li.style.setProperty('--accent', solver.color);
  li.innerHTML = `<span class="swatch" style="background:${solver.color}"></span>` +
                 `<button class="name" type="button" aria-pressed="true">${solver.name}</button>` +
                 `<span class="param"><span class="value">${config ? formatParamValue(solverParamValues.get(solver.id) ?? config.defaultValue) : '—'}</span><input type="range" aria-label="${solver.name} parameter" min="${config ? config.minExp : -4}" max="${config ? config.maxExp : 4}" step="${config ? config.step : 0.05}" value="${config ? Math.log10(solverParamValues.get(solver.id) ?? config.defaultValue) : 0}"></span>` +
                 `<span class="stats"></span>`;
  const rangeInput = li.querySelector('input');
  const valueLabel = li.querySelector('.value');
  if (rangeInput && config) {
    solverParamValues.set(solver.id, solverParamValues.get(solver.id) ?? config.defaultValue);
    rangeInput.addEventListener('input', (event) => {
      const exponent = Number(event.target.value);
      const value = 10 ** exponent;
      solverParamValues.set(solver.id, value);
      valueLabel.textContent = formatParamValue(value);
      drawSolver(solver, SOLVERS.indexOf(solver));
    });
    rangeInput.addEventListener('click', (event) => event.stopPropagation());
  }
  li.querySelector('.name').addEventListener('click', (event) => {
    if (hidden.delete(solver.id) === false) hidden.add(solver.id);
    li.classList.toggle('off', hidden.has(solver.id));
    event.currentTarget.setAttribute('aria-pressed', String(!hidden.has(solver.id)));
    const group = solverGroups.get(solver.id);
    if (group) group.visible = !hidden.has(solver.id);
  });
  $('legend').appendChild(li);
}

// A click (not a drag) on the surface picks the start point.
const raycaster = new THREE.Raycaster();
let downAt = null;
renderer.domElement.addEventListener('pointerdown', (e) => { downAt = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener('pointerup', (e) => {
  if (!downAt || e.button !== 0 || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 4)
    return;
  const ndc = new THREE.Vector2(2 * e.clientX / window.innerWidth - 1,
                                1 - 2 * e.clientY / window.innerHeight);
  raycaster.setFromCamera(ndc, camera);
  const hit = raycaster.intersectObject(surface)[0];
  if (!hit) return;
  start = [view.cx + hit.point.x / view.scale, view.cy + hit.point.z / view.scale];
  solve();
});

selectProblem(problem);
renderer.setAnimationLoop(() => {
  controls.update();
  renderer.render(scene, camera);
});
