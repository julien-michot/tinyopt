// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

// Five classical 2D least-squares problems: cost(x, y) = 0.5 * sum(r_i(x, y)^2).
// `residuals(x, y, r)` fills r, `jacobian(x, y, J)` fills the row-major numResiduals x 2 Jacobian.

export const PROBLEMS = [
  {
    id: 'rosenbrock',
    name: 'Rosenbrock',
    numResiduals: 2,
    domain: { x: [-2, 2], y: [-1, 3] },
    start: [-1.2, 1],
    minima: [[1, 1]],
    residuals: (x, y, r) => {
      r[0] = 10 * (y - x * x);
      r[1] = 1 - x;
    },
    jacobian: (x, y, J) => {
      J[0] = -20 * x; J[1] = 10;
      J[2] = -1; J[3] = 0;
    },
  },
  {
    id: 'himmelblau',
    name: 'Himmelblau',
    numResiduals: 2,
    domain: { x: [-5, 5], y: [-5, 5] },
    start: [0, 0],
    minima: [[3, 2], [-2.805118, 3.131312], [-3.779310, -3.283186], [3.584428, -1.848126]],
    residuals: (x, y, r) => {
      r[0] = x * x + y - 11;
      r[1] = x + y * y - 7;
    },
    jacobian: (x, y, J) => {
      J[0] = 2 * x; J[1] = 1;
      J[2] = 1; J[3] = 2 * y;
    },
  },
  {
    id: 'beale',
    name: 'Beale',
    numResiduals: 3,
    domain: { x: [-4.5, 4.5], y: [-4.5, 4.5] },
    start: [1, 1],
    minima: [[3, 0.5]],
    residuals: (x, y, r) => {
      r[0] = 1.5 - x + x * y;
      r[1] = 2.25 - x + x * y * y;
      r[2] = 2.625 - x + x * y * y * y;
    },
    jacobian: (x, y, J) => {
      J[0] = y - 1; J[1] = x;
      J[2] = y * y - 1; J[3] = 2 * x * y;
      J[4] = y * y * y - 1; J[5] = 3 * x * y * y;
    },
  },
  {
    id: 'booth',
    name: 'Booth',
    numResiduals: 2,
    domain: { x: [-10, 10], y: [-10, 10] },
    start: [-8, 7],
    minima: [[1, 3]],
    residuals: (x, y, r) => {
      r[0] = x + 2 * y - 7;
      r[1] = 2 * x + y - 5;
    },
    jacobian: (x, y, J) => {
      J[0] = 1; J[1] = 2;
      J[2] = 2; J[3] = 1;
    },
  },
  {
    id: 'freudenstein-roth',
    name: 'Freudenstein-Roth',
    numResiduals: 2,
    domain: { x: [-5, 20], y: [-6, 8] },
    start: [0.5, -2],
    minima: [[5, 4], [11.412779, -0.896805]], // The second one is only a local minimum.
    residuals: (x, y, r) => {
      r[0] = -13 + x + ((5 - y) * y - 2) * y;
      r[1] = -29 + x + ((y + 1) * y - 14) * y;
    },
    jacobian: (x, y, J) => {
      J[0] = 1; J[1] = 10 * y - 3 * y * y - 2;
      J[2] = 1; J[3] = 3 * y * y + 2 * y - 14;
    },
  },
];

const scratch = new Float64Array(8);

export function cost(problem, x, y) {
  problem.residuals(x, y, scratch);
  let c = 0;
  for (let i = 0; i < problem.numResiduals; ++i) c += scratch[i] * scratch[i];
  return 0.5 * c;
}
