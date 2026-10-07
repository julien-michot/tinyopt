// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/optimizers/gd.h>
#include <tinyopt/tinyopt.h>

tinyopt::Vector<double, 1> PrecompiledResiduals(const double &parameter);
using PrecompiledResidualFunction = decltype(&PrecompiledResiduals);

extern template tinyopt::Summary tinyopt::Optimize<double, PrecompiledResidualFunction>(
    double &parameter, const PrecompiledResidualFunction &residuals, const tinyopt::Options &);

extern template class tinyopt::gd::Optimizer<tinyopt::Vec1f>;