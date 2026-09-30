// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>
#include <utility>

namespace tinyopt {

template <typename... Ts>
struct ParamsPack {
  std::tuple<Ts...> values;

  explicit ParamsPack(Ts... params) : values(std::forward<Ts>(params)...) {}
};

template <typename... Ts>
ParamsPack(Ts &...) -> ParamsPack<Ts &...>;

}  // namespace tinyopt