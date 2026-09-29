// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/tinyopt.h>

using ResidualFunction = tinyopt::Vector<double, 1> (*)(const double &);

tinyopt::Vector<double, 1> Residuals(const double &parameter);

extern template tinyopt::Output tinyopt::Optimize<double, ResidualFunction>(
    double &parameter, const ResidualFunction &residuals, const tinyopt::Options &);