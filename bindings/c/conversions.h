#ifndef TINYOPT_CONVERSIONS_H
#define TINYOPT_CONVERSIONS_H

#ifdef __cplusplus
#include <tinyopt/optimizers/options.h>
#include <tinyopt/output.h>
#endif

#ifdef __cplusplus
namespace tinyopt {
struct Options;
struct Output;
}  // namespace tinyopt
// C++ prototypes for conversion helpers implemented in bindings/generated_bindings.cpp
tinyopt::Options Convert(const options_t &in);
output_t Convert(const tinyopt::Output &in);
#endif

#endif  // TINYOPT_CONVERSIONS_H
