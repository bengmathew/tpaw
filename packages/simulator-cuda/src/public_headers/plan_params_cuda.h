#ifndef PLAN_PARAMS_CUDA_H
#define PLAN_PARAMS_CUDA_H

#include "historical_returns_cuda.h"
#include "numeric_types.h"
#include "opt_currency.h"
#include "stocks_and_bonds_float.h"
#include <stdint.h> // bindgen cannot find cstdint (needed for sized number types).

// bindgen (to generate rust bindings) has trouble with extern "C"
// because it is expecting C not C++ and extern "C" is a C++ thing.
#ifdef __cplusplus
extern "C" {
#endif

struct PlanParamsCuda {
  struct Ages {
    struct SimulationMonths {
      uint32_t num_months;
      uint32_t withdrawal_start_month;
    };
    struct SimulationMonths simulation_months;
  };

  struct AdjustmentsToSpending {
    struct TPAWAndSPAW {
      struct OptCURRENCY spending_ceiling;
      struct OptCURRENCY spending_floor;
      CURRENCY legacy;
    };
    struct TPAWAndSPAW tpaw_and_spaw;
  };

  struct Risk {
    struct TPAW {
      FLOAT time_preference;
      FLOAT annual_additional_spending_tilt;
      FLOAT legacy_rra_including_pos_infinity;
    };
    struct SWR {
      enum WithdrawalType { Percent, Amount };
      enum WithdrawalType withdrawal_type;
      CURRENCY withdrawal_as_percent_or_amount;
    };
    struct TPAW tpaw;
    struct SWR swr;
  };

  struct Advanced {
    struct ReturnStatsForPlanning {
      struct StocksAndBondsFLOAT expected_returns_at_month_0;
      FLOAT annual_empirical_log_variance_stocks;
    };
    struct Sampling {
      enum Type { MonteCarloSampling, HistoricalSampling };
      struct MonteCarlo {
        uint64_t seed;
        uint32_t num_runs;
        uint32_t block_size;
        uint32_t stagger_run_starts;
      };
      struct Historical {
        // Needed because empty struct has different size in C and C++!
        uint32_t ignore;
      };
      union MonteCarloOrHistorical {
        struct MonteCarlo monte_carlo;
        struct Historical historical;
      };

      enum Type type;
      union MonteCarloOrHistorical monte_carlo_or_historical;
    };
    enum Strategy { Strategy_TPAW, Strategy_SPAW, Strategy_SWR };

    struct ReturnStatsForPlanning return_stats_for_planning;
    struct Sampling sampling;
    enum Strategy strategy;
  };

  struct Ages ages;
  struct AdjustmentsToSpending adjustments_to_spending;
  struct Risk risk;
  struct Advanced advanced;
};

// We break arrays out into a separate stuct because we want to convert all the
// c-style arrays to vectors right at the rust-C bridge. Interspersing them in
// the PlanParamsCuda struct means we will have to redefine the whole struct.
// Having it as a separate struct restricts the amount of redefinition needed.
struct PlanParamsCuda_C_Arrays {
  CURRENCY *future_savings_by_mfn;
  CURRENCY *income_during_retirement_by_mfn;
  CURRENCY *essential_expenses_by_mfn;
  CURRENCY *discretionary_expenses_by_mfn;
  FLOAT *tpaw_rra_including_pos_infinity_by_mfn;
  FLOAT *spaw_spending_tilt_by_mfn;
  FLOAT *spaw_and_swr_stock_allocation_savings_portfolio_by_mfn;
  uint32_t num_percentiles;
  uint32_t *percentiles;
  uint32_t historical_returns_series_len;
  struct HistoricalReturnsCuda *historical_returns_series;
};

#ifdef __cplusplus
}
#endif

#endif // PLAN_PARAMS_CUDA_H
