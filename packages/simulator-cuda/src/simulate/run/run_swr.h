
#ifndef RUN_SWR_H
#define RUN_SWR_H

#include "src/public_headers/numeric_types.h"
#include "src/public_headers/stocks_and_bonds_float.h"
#include "src/simulate/cuda_process_run_x_mfn_simulated_x_mfn/cuda_process_swr_run_x_mfn_simulated_x_mfn.h"
#include "src/simulate/run/run_result_padded.h"

namespace swr {
  RunResultPadded
  run(const uint32_t num_runs,
      const uint32_t num_months_to_simulate,
      const CURRENCY current_portfolio_balance,
      const uint32_t withdrawal_start_month,
      const PlanParamsCuda::Risk::SWR::WithdrawalType withdrawal_type,
      const CURRENCY withdrawal_as_percent_or_amount,
      const thrust::device_vector<CURRENCY> &income_by_mfn,
      const thrust::device_vector<CURRENCY> &essential_expense_by_mfn,
      const thrust::device_vector<CURRENCY> &discretionary_expense_by_mfn,
      const thrust::device_vector<FLOAT>
          &stock_allocation_savings_portfolio_by_mfn,
      const thrust::device_vector<StocksAndBondsFLOAT>
          &historical_returns_by_run_by_mfn_simulated,
      const Cuda_Processed_SWR_Run_x_MFNSimulated_x_MFN
          &cuda_processed_run_x_mfn_simulated_x_mfn);
} // namespace swr
  //
#endif // RUN_SWR_H