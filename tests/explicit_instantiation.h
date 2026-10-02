// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/tinyopt.h>

struct Quadratic {
  template <typename Scalar>
  Scalar operator()(const Scalar &value) const {
    const Scalar residual = value - Scalar(2);
    return residual * residual;
  }
};

extern template tinyopt::Output tinyopt::Optimize<double, Quadratic>(
    double &parameter, const Quadratic &cost, const tinyopt::Options &);