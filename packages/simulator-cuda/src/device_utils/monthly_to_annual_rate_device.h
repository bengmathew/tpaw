#ifndef MONTHLY_TO_ANNUAL_RATE_DEVICE_H
#define MONTHLY_TO_ANNUAL_RATE_DEVICE_H

#include "src/public_headers/numeric_types.h"

__device__ __forceinline__ FLOAT
monthly_to_annual_rate_device(const FLOAT monthly) {
  return __FLOAT_POWF(FLOAT_L(1.0) + monthly, FLOAT_L(12.0)) - FLOAT_L(1.0);
}

#endif // MONTHLY_TO_ANNUAL_RATE_DEVICE_H
