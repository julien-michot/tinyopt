// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include "explicit_instantiation.h"

template tinyopt::Output tinyopt::Optimize<double, Quadratic>(
    double &parameter, const Quadratic &cost, const tinyopt::Options &);