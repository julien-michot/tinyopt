// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/losses/helpers.h>

namespace tinyopt::losses {

/**
 * @name Activation losses
 * @{
 */
/// @brief Sigmoid = 1/(1+e^-x), derivative = Sigmoid(x) * (1 - Sigmoid(x))
DefineLoss(
    Sigmoid, 1.0 / (1.0 + exp(-x)),
    l.unaryExpr([](auto x) -> Scalar { return x * (Scalar(1.0) - x); }).reshaped().asDiagonal());

/// @brief Tanh = (e^x-e^-x)/(e^x+e^-x),   derivative = 1 - Tanh(x)^2
DefineLoss(
    Tanh, (exp(x) - exp(-x)) / (exp(x) + exp(-x)),
    l.unaryExpr([](auto x) -> Scalar { return Scalar(1.0) - x * x; }).reshaped().asDiagonal());

/// @brief ReLU = max(0, x),  derivative = {x>0:1, x<=0:0}
DefineLoss(
    ReLU, x > 0 ? x : Scalar(0.0),
    x.unaryExpr([](auto x) { return x > 0 ? Scalar(1.0) : Scalar(0.0); }).reshaped().asDiagonal());

/// @brief LeakyReLU = {x>0:x, x<=0:a*x}, derivative = {x>0:1, x<=0:a}
DefineLoss2(LeakyReLU, x > 0 ? x : a * x,
            x.unaryExpr([a](auto x) { return x > 0 ? 1 : a; }).reshaped().asDiagonal());

/** @} */
}  // namespace tinyopt::losses
