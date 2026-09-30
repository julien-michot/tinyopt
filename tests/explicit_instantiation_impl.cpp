// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include "explicit_instantiation.h"

tinyopt::Vector<double, 1> PrecompiledResiduals(const double &parameter) {
    return tinyopt::Vector<double, 1>(parameter - 2.0);
}

template tinyopt::Output tinyopt::Optimize<double, decltype(&PrecompiledResiduals)>(
        double &parameter, const decltype(&PrecompiledResiduals) &residuals, const tinyopt::Options &);