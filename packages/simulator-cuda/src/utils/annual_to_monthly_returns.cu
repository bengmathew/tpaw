#include "annual_to_monthly_returns.h"

FLOAT annual_to_monthly_return(const FLOAT annual_return) {
  return std::pow(1.0 + annual_return, 1.0 / 12.0) - 1.0;
}