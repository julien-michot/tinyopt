// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include "explicit_instantiation.h"

tinyopt::Vector<double, 1> Residuals(const double &parameter) {
    return tinyopt::Vector<double, 1>(parameter - 2.0);
}

template tinyopt::Output tinyopt::Optimize<double, ResidualFunction>(
        double &parameter, const ResidualFunction &residuals, const tinyopt::Options &);