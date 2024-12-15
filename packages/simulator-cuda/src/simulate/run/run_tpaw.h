
#ifndef RUN_TPAW_H
#define RUN_TPAW_H

#include "src/public_headers/numeric_types.h"
#include "src/public_headers/opt_currency.h"
#include "src/public_headers/stocks_and_bonds_float.h"
#include "src/simulate/cuda_process_run_x_mfn_simulated_x_mfn/cuda_process_tpaw_run_x_mfn_simulated_x_mfn.h"
#include "src/simulate/run/run_result_padded.h"

namespace tpaw {
  RunResultPadded
  run(const uint32_t num_runs_in,
      const uint32_t num_months_to_simulate,
      const CURRENCY current_portfolio_balance,
      const uint32_t withdrawal_start_month,
      const OptCURRENCY spending_ceiling,
      const OptCURRENCY spending_floor,
      const StocksAndBondsFLOAT expected_returns,
      const thrust::device_vector<CURRENCY> &income_by_mfn,
      const thrust::device_vector<CURRENCY> &essential_expense_by_mfn,
      const thrust::device_vector<CURRENCY> &discretionary_expense_by_mfn,
      const thrust::device_vector<StocksAndBondsFLOAT>
          &historical_returns_by_run_by_mfn_simulated,
      const Cuda_Processed_TPAW_Run_x_MFNSimulated_x_MFN
          &cuda_processed_run_x_mfn_simulated_x_mfn);
} // namespace _tpaw
#endif // RUN_TPAW_H