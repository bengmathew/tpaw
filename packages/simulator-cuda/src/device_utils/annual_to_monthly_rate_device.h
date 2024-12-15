#ifndef ANNUAL_TO_MONTHLY_RATE_DEVICE_H
#define ANNUAL_TO_MONTHLY_RATE_DEVICE_H

#include "src/public_headers/numeric_types.h"

__device__ __forceinline__ FLOAT
annual_to_monthly_rate_device(const FLOAT annual) {
  return __FLOAT_POWF(FLOAT_L(1.0) + annual,
                      FLOAT_DIVIDE(FLOAT_L(1.0), FLOAT_L(12.0))) -
         FLOAT_L(1.0);
}

#endif // ANNUAL_TO_MONTHLY_RATE_DEVICE_H
