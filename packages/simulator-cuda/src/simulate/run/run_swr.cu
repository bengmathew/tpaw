#include "run_swr.h"
#include "src/public_headers/numeric_types.h"
#include "src/public_headers/opt_currency.h"
#include "src/simulate/run/run_common.h"
#include "src/utils/run_mfn_indexing.h"
#include <cstdint>
#include <cstdio>

namespace swr {

  // *****************************************************************************
  // _get_target_withdrawals_assuming_no_ceiling_or_floor
  // *****************************************************************************
  namespace _get_target_withdrawals_assuming_no_ceiling_or_floor {
    __device__ __forceinline__ run_common::TargetWithdrawals
    fn(const uint32_t month_index,
       const uint32_t withdrawal_start_month,
       const CURRENCY balance_starting,
       const CURRENCY prev_target_withdrawal_general,
       const CURRENCY current_month_essential_expense,
       const CURRENCY current_month_discretionary_expense,
       const PlanParamsCuda::Risk::SWR::WithdrawalType withdrawal_type,
       const CURRENCY withdrawal_as_percent_or_amount) {
      const CURRENCY general =
          month_index < withdrawal_start_month ? 0.0
          : month_index > withdrawal_start_month
              ? prev_target_withdrawal_general
          : withdrawal_type ==
                  PlanParamsCuda::Risk::SWR::WithdrawalType::Percent
              // Note, balance_starting when month_index ==
              // withdrawal_start_month is the balance at the start of
              // retirement.
              ? fmax(balance_starting * withdrawal_as_percent_or_amount, 0.0)
              : withdrawal_as_percent_or_amount;

      return {
          .essential = current_month_essential_expense,
          .discretionary = current_month_discretionary_expense,
          .general = general,
      };
    }
  } // namespace _get_target_withdrawals_assuming_no_ceiling_or_floor

  // *****************************************************************************
  // _single_month
  // *****************************************************************************
  namespace _single_month {
    struct _ReturnValue {
      FLOAT withdrawals_from_savings_portfolio_rate;
      CURRENCY balance_ending;
      CURRENCY target_withdrawal_general;

      __host__ __device__ void print(uint32_t num_tabs) const {
        const auto i = static_cast<int32_t>(num_tabs * 4);
        printf("%*s"
               "withdrawals_from_savings_portfolio_rate: %.65f\n",
               i,
               "",
               withdrawals_from_savings_portfolio_rate);
        printf("%*s"
               "balance_ending: %.65f\n",
               i,
               "",
               balance_ending);
        printf("%*s"
               "target_withdrawal_general: %.65f\n",
               i,
               "",
               target_withdrawal_general);
      }
    };

    template <bool write_result>
    __device__ __forceinline__ _ReturnValue
    fn(const uint32_t run_index,
       const uint32_t month_index,
       const CURRENCY balance_starting,
       const CURRENCY prev_target_withdrawal_general,
       const uint32_t withdrawal_start_month,
       const PlanParamsCuda::Risk::SWR::WithdrawalType withdrawal_type,
       const CURRENCY withdrawal_as_percent_or_amount,
       const CURRENCY current_month_income,
       const CURRENCY current_month_essential_expense,
       const CURRENCY current_month_discretionary_expense,
       const FLOAT current_month_stock_allocation_savings_portfolio,
       const StocksAndBondsFLOAT &returns,
       const Cuda_Processed_SWR_Run_x_MFNSimulated_x_MFN::Entry
           &cuda_processed_run_x_mfn_simulated_x_mfn,
       RunResultPadded_GPU *run_result_padded) {

      // Step 1: Precomputation at start
      const bool withdrawal_started = month_index >= withdrawal_start_month;

      // Step 2: Data for expected run (if needed).
      // Not needed for SWR.

      // Step 3: Calculate target withdrawals before ceiling and floor (note
      // this  happens before contributions are applied).
      const run_common::TargetWithdrawals
          target_withdrawals_before_ceiling_and_floor =
              _get_target_withdrawals_assuming_no_ceiling_or_floor::fn(
                  month_index,
                  withdrawal_start_month,
                  balance_starting,
                  prev_target_withdrawal_general,
                  current_month_essential_expense,
                  current_month_discretionary_expense,
                  withdrawal_type,
                  withdrawal_as_percent_or_amount);

      // Step 4: Apply ceiling and floor to target withdrawals
      const run_common::TargetWithdrawals target_withdrawals =
          run_common::apply_withdrawal_ceiling_and_floor(
              target_withdrawals_before_ceiling_and_floor,
              // Note: this should NOT be the scaled discretionary expense.
              current_month_discretionary_expense,
              // No ceiling or floor for SWR.
              OptCURRENCY{.is_set = false, .opt_value = 0.0},
              OptCURRENCY{.is_set = false, .opt_value = 0.0},
              withdrawal_started);

      // Step 5: Apply contributions and withdrawals
      const run_common::AfterContributionsAndWithdrawals
          savings_portfolio_after_contributions_and_withdrawals =
              run_common::apply_contributions_and_withdrawals(
                  balance_starting, current_month_income, target_withdrawals);

      // Step 5.5: Handle NaN withdrawals from savings portfolio rate.
      const FLOAT withdrawals_from_savings_portfolio_rate =
          (!isfinite(savings_portfolio_after_contributions_and_withdrawals
                         .withdrawals.from_savings_portfolio_rate_or_nan_or_inf)
               ? ({
                   fn<false>(run_index,
                             month_index,
                             0.000001,
                             prev_target_withdrawal_general,
                             withdrawal_start_month,
                             withdrawal_type,
                             withdrawal_as_percent_or_amount,
                             current_month_income,
                             current_month_essential_expense,
                             current_month_discretionary_expense,
                             current_month_stock_allocation_savings_portfolio,
                             returns,
                             cuda_processed_run_x_mfn_simulated_x_mfn,
                             run_result_padded)
                       .withdrawals_from_savings_portfolio_rate;
                 })
               : savings_portfolio_after_contributions_and_withdrawals
                     .withdrawals.from_savings_portfolio_rate_or_nan_or_inf);

      // Step 6: Calculate stock allocation
      const FLOAT stock_allocation =
          current_month_stock_allocation_savings_portfolio;

      // Step 7: Apply allocation
      const run_common::End savings_portfolio_at_end =
          run_common::apply_allocation(
              stock_allocation,
              returns,
              savings_portfolio_after_contributions_and_withdrawals.balance,
              cuda_processed_run_x_mfn_simulated_x_mfn.npv_approx
                  .income_bond_rate_without_current_month);

      // Step 8: Write results (in write_result)
      if (write_result) {
        run_common::write_result(
            balance_starting,
            savings_portfolio_after_contributions_and_withdrawals,
            withdrawals_from_savings_portfolio_rate,
            savings_portfolio_at_end,
            0.0, // tpaw_spending_tilt
            *run_result_padded,
            run_index,
            month_index);
      }

      // Step 9 :Return
      return _ReturnValue{
          .withdrawals_from_savings_portfolio_rate =
              withdrawals_from_savings_portfolio_rate,
          .balance_ending = savings_portfolio_at_end.balance,
          .target_withdrawal_general = target_withdrawals.general,
      };
    }
  } // namespace _single_month

  // *****************************************************************************
  // _kernel
  // *****************************************************************************
  __global__ void
  _kernel(const uint32_t num_runs,
          const uint32_t num_months_to_simulate,
          const CURRENCY current_portfolio_balance,
          const uint32_t withdrawal_start_month,
          const PlanParamsCuda::Risk::SWR::WithdrawalType withdrawal_type,
          const CURRENCY withdrawal_as_percent_or_amount,
          const CURRENCY *const income_by_mfn,
          const CURRENCY *const essential_expense_by_mfn,
          const CURRENCY *const discretionary_expense_by_mfn,
          const FLOAT *const stock_allocation_savings_portfolio_by_mfn,
          const StocksAndBondsFLOAT
              *const historical_returns_by_run_by_mfn_simulated,
          const Cuda_Processed_SWR_Run_x_MFNSimulated_x_MFN::Entry
              *const cuda_processed_run_x_mfn_simulated_x_mfn,
          RunResultPadded_GPU *const run_result_padded) {
    const uint32_t run_index = (threadIdx.x + blockIdx.x * blockDim.x);
    if (run_index >= num_runs)
      return;

    CURRENCY balance_starting = current_portfolio_balance;
    CURRENCY prev_target_withdrawal_general = 0.0;
    for (uint32_t month_index = 0; month_index < num_months_to_simulate;
         month_index++) {
      const uint32_t run_by_mfn_simulated_index = get_run_by_mfn_index(
          num_runs, num_months_to_simulate, run_index, month_index);
      const _single_month::_ReturnValue current_month_result =
          _single_month::fn<true>(
              run_index,
              month_index,
              balance_starting,
              prev_target_withdrawal_general,
              withdrawal_start_month,
              withdrawal_type,
              withdrawal_as_percent_or_amount,
              income_by_mfn[month_index],
              essential_expense_by_mfn[month_index],
              discretionary_expense_by_mfn[month_index],
              stock_allocation_savings_portfolio_by_mfn[month_index],
              historical_returns_by_run_by_mfn_simulated
                  [run_by_mfn_simulated_index],
              cuda_processed_run_x_mfn_simulated_x_mfn
                  [run_by_mfn_simulated_index],
              run_result_padded);
      balance_starting = current_month_result.balance_ending;
      prev_target_withdrawal_general =
          current_month_result.target_withdrawal_general;
    }
  }

  //  *****************************************************************************
  // run
  // *****************************************************************************

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
          &cuda_processed_run_x_mfn_simulated_x_mfn) {

    RunResultPadded run_result_padded_host_struct =
        RunResultPadded::make(num_runs, num_months_to_simulate);
    unique_ptr_gpu<RunResultPadded_GPU> run_result_padded_device_struct =
        run_result_padded_host_struct.copy_to_gpu();

    const int32_t block_size = 64;
    _kernel<<<(num_runs + block_size - 1) / block_size, block_size>>>(
        num_runs,
        num_months_to_simulate,
        current_portfolio_balance,
        withdrawal_start_month,
        withdrawal_type,
        withdrawal_as_percent_or_amount,
        income_by_mfn.data().get(),
        essential_expense_by_mfn.data().get(),
        discretionary_expense_by_mfn.data().get(),
        stock_allocation_savings_portfolio_by_mfn.data().get(),
        historical_returns_by_run_by_mfn_simulated.data().get(),
        cuda_processed_run_x_mfn_simulated_x_mfn.for_normal_run.data().get(),
        run_result_padded_device_struct.get());
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return run_result_padded_host_struct;
  }

} // namespace swr
