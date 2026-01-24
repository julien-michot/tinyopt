// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/3rdparty/ceres/jet.h>  // should not be another one
#include <tinyopt/math.h>

namespace tinyopt::diff {

/// The Automatix differentiation Jet struct
template <typename T, int N>
using Jet = ceres::Jet<T, N>;

}  // namespace tinyopt::diff

#include <tinyopt/diff/jet_traits.h>