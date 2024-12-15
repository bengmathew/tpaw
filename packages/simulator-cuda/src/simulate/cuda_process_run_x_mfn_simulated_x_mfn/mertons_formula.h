
#ifndef MERTONS_FORMULA_H
#define MERTONS_FORMULA_H

#include "src/device_utils/annual_to_monthly_rate_device.h"
#include "src/public_headers/numeric_types.h"
#include <cmath>
#include <cstdio>
#include <stdint.h> // bindgen cannot find cstdint (needed for sized number types).

struct MertonsFormulaResult {
  FLOAT stock_allocation;
  FLOAT spending_tilt;

  __device__ __host__ void print(uint32_t num_tabs) const;
};

struct PlainMertonsFormulaClosure {
  FLOAT annual_equity_premium_by_variance;
  FLOAT c0;
  FLOAT c1;
  FLOAT annual_additional_spending_tilt;

  __host__ __device__ void print(uint32_t num_tabs) const;
};

// The function may be more general, but we ensure correctness only when:
// 1. annual_equity_premium >= 0
// 3. annual_variance_stocks > 0
__device__ __forceinline__ PlainMertonsFormulaClosure
get_plain_mertons_formula_closure(const FLOAT annual_r, // bond rate,
                                  const FLOAT annual_equity_premium,
                                  const FLOAT annual_variance_stocks,
                                  const FLOAT time_preference,
                                  const FLOAT annual_additional_spending_tilt) {
  const FLOAT rho = time_preference;
  const FLOAT annual_equity_premium_by_variance_stocks =
      __FLOAT_DIVIDE(annual_equity_premium, annual_variance_stocks);
  const FLOAT annual_equity_premium_pow2_by_2variance_stocks =
      annual_equity_premium * annual_equity_premium_by_variance_stocks *
      FLOAT_L(0.5);

  const PlainMertonsFormulaClosure result{
      .annual_equity_premium_by_variance =
          annual_equity_premium_by_variance_stocks,
      .c0 = annual_r - rho + annual_equity_premium_pow2_by_2variance_stocks,
      .c1 = annual_equity_premium_pow2_by_2variance_stocks,
      .annual_additional_spending_tilt = annual_additional_spending_tilt,
  };
  return result;
}

// The function may be more general, but we ensure correctness only when:
// 1. rra_including_pos_infinity > 0
__device__ __forceinline__ MertonsFormulaResult
plain_mertons_formula(const PlainMertonsFormulaClosure &closure,
                      const FLOAT rra_including_pos_infinity) {
  if (rra_including_pos_infinity == INFINITY) {
    return MertonsFormulaResult{
        .stock_allocation = FLOAT_L(0.0),
        .spending_tilt = annual_to_monthly_rate_device(
            closure.annual_additional_spending_tilt),
    };
  }

  const FLOAT gamma = rra_including_pos_infinity;
  const FLOAT one_over_gamma = __FLOAT_DIVIDE(FLOAT_L(1.0), gamma);
  const FLOAT one_over_gamma_pow2 = one_over_gamma * one_over_gamma;

  const FLOAT stock_allocation =
      closure.annual_equity_premium_by_variance * one_over_gamma;

  const FLOAT annual_spending_tilt =
      FLOAT_MA(one_over_gamma,
               closure.c0,
               FLOAT_MA(one_over_gamma_pow2,
                        closure.c1,
                        closure.annual_additional_spending_tilt));

  return MertonsFormulaResult{
      .stock_allocation = stock_allocation,
      .spending_tilt = annual_to_monthly_rate_device(annual_spending_tilt)};
}

__device__ __forceinline__ FLOAT get_rra_for_all_stocks(
    const FLOAT annual_equity_premium, const FLOAT annual_sigma_pow2) {
  return __FLOAT_DIVIDE(annual_equity_premium, annual_sigma_pow2);
}

struct EffectiveMertonsFormulaClosure {
  PlainMertonsFormulaClosure plain_closure;
  FLOAT rra_for_all_stocks;

  __host__ __device__ void print(uint32_t num_tabs) const;
};

__device__ __forceinline__ EffectiveMertonsFormulaClosure
get_effective_mertons_formula_closure(
    const FLOAT annual_r, // bond rate,
    const FLOAT annual_equity_premium,
    const FLOAT annual_variance_stocks,
    const FLOAT time_preference,
    const FLOAT annual_additional_spending_tilt) {
  // When equity premium is < 0, Merton's formula yields a stock allocation of
  // -Infinity and 0 for rra of 0 and Infinity respectively. But we don't really
  // want to handle this case using Merton's formula, because negative stock
  // allocation means leverage, which we don't allow. We want instead to
  // completely ignore stocks. We do this by bringing equity premium to 0.
  const FLOAT annual_effective_equity_premium =
      FLOAT_MAX(FLOAT_L(0.0), annual_equity_premium);

  return EffectiveMertonsFormulaClosure{
      .plain_closure =
          get_plain_mertons_formula_closure(annual_r,
                                            annual_effective_equity_premium,
                                            annual_variance_stocks,
                                            time_preference,
                                            annual_additional_spending_tilt),
      .rra_for_all_stocks = get_rra_for_all_stocks(
          annual_effective_equity_premium, annual_variance_stocks),
  };
}

// Different from plain_mertons_formula due to handling of equity premium ranges
// and clamping rra.
__device__ __forceinline__ MertonsFormulaResult
effective_mertons_formula(const EffectiveMertonsFormulaClosure &closure,
                          const FLOAT rra_including_pos_infinity) {

  // Clamp rra to 100% stocks.
  const FLOAT effective_rra =
      FLOAT_MAX(closure.rra_for_all_stocks, rra_including_pos_infinity);

  MertonsFormulaResult result =
      plain_mertons_formula(closure.plain_closure, effective_rra);

  // effective_rra nominally clamps to 100% stocks, but there can
  // be floating point imprecision.
  result.stock_allocation = __FLOAT_SATURATE(result.stock_allocation);
  return result;
}

__device__ __forceinline__ FLOAT
effective_mertons_formula_stock_allocation_only(
    const FLOAT annual_equity_premium,
    const FLOAT annual_variance_stocks, // variance of stock return
    const FLOAT rra_including_pos_infinity) {
  const EffectiveMertonsFormulaClosure closure =
      get_effective_mertons_formula_closure(FLOAT_L(0.0), // not used
                                            annual_equity_premium,
                                            annual_variance_stocks,
                                            FLOAT_L(0.0), // not used
                                            FLOAT_L(0.0)  // not used
      );
  return effective_mertons_formula(closure, rra_including_pos_infinity)
      .stock_allocation;
}

#endif // MERTONS_FORMULA_H
