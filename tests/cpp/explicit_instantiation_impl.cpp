// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <tinyopt/optimizers/gd.h>
#include "explicit_instantiation.h"

// These are used by simple.cpp tests.

tinyopt::Vector<double, 1> PrecompiledResiduals(const double &parameter) {
  return tinyopt::Vector<double, 1>(parameter - 2.0);
}

template tinyopt::Summary tinyopt::Optimize<double, decltype(&PrecompiledResiduals)>(
    double &parameter, const decltype(&PrecompiledResiduals) &residuals, const tinyopt::Options &);

// Also instantiate a specific Optimizer
template class tinyopt::gd::Optimizer<tinyopt::Vec1f>;