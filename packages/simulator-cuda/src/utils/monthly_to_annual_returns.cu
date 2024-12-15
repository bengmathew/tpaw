#include "annual_to_monthly_returns.h"

FLOAT monthly_to_annual_return(const FLOAT monthly_return) {
  return std::pow(1.0 + monthly_return, 12.0) - 1.0;
}