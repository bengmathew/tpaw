#ifndef HISTORICAL_RETURNS_CUDA_H
#define HISTORICAL_RETURNS_CUDA_H

#include "numeric_types.h"

#ifdef __cplusplus
extern "C" {
#endif

struct HistoricalReturnsCuda {
  struct Part {
    FLOAT returns;
    // TODO: When implementing duration matching, make sure these changes can be
    // applied to the annual expected returns expressed as monthly, which in the
    // unit for the expected return input in PlanParamsCuda.
    FLOAT expected_return_change;
  };

  struct Part stocks;
  struct Part bonds;
};

#ifdef __cplusplus
}
#endif

#endif // HISTORICAL_RETURNS_CUDA_H
