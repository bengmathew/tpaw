#include "run_spaw.h"
#include "src/public_headers/numeric_types.h"
#include "src/simulate/run/run_common.h"
#include "src/utils/run_mfn_indexing.h"
#include <cstdint>
#include <cstdio>

namespace spaw {
  struct _ExpectedRunData {
    CURRENCY wealth_minus_npv_essential_at_start;
    __device__ __host__ void print(uint32_t num_tabs) const;
  };

  __device__ __host__ void _ExpectedRunData::print(uint32_t num_tabs) const {
    const auto i = static_cast<int32_t>(num_tabs * 4);
    printf("%*swealth_minus_npv_essential_at_start: %.65f\n",
           i,
           "",
           wealth_minus_npv_essential_at_start);
  }

  // *****************************************************************************
  // _get_precomputation_at_start
  // *****************************************************************************
  struct _PrecomputationAtStart {
    CURRENCY npv_approx_income_bond_rate_with_current_month;
    CURRENCY npv_approx_income_portfolio_rate_with_current_month;
    CURRENCY npv_approx_essential_expenses_with_current_month;
    CURRENCY npv_approx_discretionary_expenses_with_current_month;
    CURRENCY wealth_minus_npv_essential_at_start;
    __device__ __host__ void print(uint32_t num_tabs) const {
      const auto i = static_cast<int32_t>(num_tabs * 4);
      printf("%*s"
             "npv_approx_income_bond_rate_with_current_month: %.65f\n",
             i,
             "",
             npv_approx_income_bond_rate_with_current_month);
      printf("%*s"
             "npv_approx_income_portfolio_rate_with_current_month: %.65f\n",
             i,
             "",
             npv_approx_income_portfolio_rate_with_current_month);
      printf("%*s"
             "npv_approx_essential_expenses_with_current_month: %.65f\n",
             i,
             "",
             npv_approx_essential_expenses_with_current_month);
      printf("%*s"
             "npv_approx_discretionary_expenses_with_current_month: %.65f\n",
             i,
             "",
             npv_approx_discretionary_expenses_with_current_month);
      printf("%*s"
             "wealth_minus_npv_essential_at_start: %.65f\n",
             i,
             "",
             wealth_minus_npv_essential_at_start);
    }
  };
  namespace _get_precomputation_at_start {

    __device__ __forceinline__ _PrecomputationAtStart
    fn(const CURRENCY balance_starting,
       const CURRENCY current_month_income,
       const CURRENCY current_month_essential_expense,
       const CURRENCY current_month_discretionary_expense,
       const CURRENCY npv_approx_income_bond_rate_without_current_month,
       const CURRENCY npv_approx_income_portfolio_rate_without_current_month,
       const CURRENCY npv_approx_essential_expenses_without_current_month,
       const CURRENCY npv_approx_discretionary_expenses_without_current_month,
       const bool debug) {

      const CURRENCY npv_approx_income_bond_rate_with_current_month =
          npv_approx_income_bond_rate_without_current_month +
          current_month_income;
      const CURRENCY npv_approx_income_portfolio_rate_with_current_month =
          npv_approx_income_portfolio_rate_without_current_month +
          current_month_income;
      const CURRENCY npv_approx_essential_expenses_with_current_month =
          npv_approx_essential_expenses_without_current_month +
          current_month_essential_expense;
      return _PrecomputationAtStart{
          .npv_approx_income_bond_rate_with_current_month =
              npv_approx_income_bond_rate_with_current_month,
          .npv_approx_income_portfolio_rate_with_current_month =
              npv_approx_income_portfolio_rate_with_current_month,
          .npv_approx_essential_expenses_with_current_month =
              npv_approx_essential_expenses_with_current_month,
          .npv_approx_discretionary_expenses_with_current_month =
              npv_approx_discretionary_expenses_without_current_month +
              current_month_discretionary_expense,
          .wealth_minus_npv_essential_at_start =
              balance_starting +
              // Intentionally using portfolio_rate and not bond_rate.
              npv_approx_income_portfolio_rate_with_current_month -
              npv_approx_essential_expenses_with_current_month,
      };
    }
  } // namespace _get_precomputation_at_start
  // *****************************************************************************
  // _get_target_withdrawals_assuming_no_ceiling_or_floor
  // *****************************************************************************
  namespace _get_target_withdrawals_assuming_no_ceiling_or_floor {
    template <bool is_expected_run>
    __device__ __forceinline__ run_common::TargetWithdrawals
    fn(const bool withdrawal_started,
       const FLOAT curr_month_cumulative_1_plus_g_over_1_plus_r,
       const CURRENCY wealth_minus_npv_essential_at_start,
       const CURRENCY npv_approx_discretionary_with_current_month,
       const CURRENCY npv_approx_legacy,
       const CURRENCY current_month_essential_expense,
       const CURRENCY current_month_discretionary_expense,
       const _ExpectedRunData
           *const expected_run_data, // nullptr if is_expected_run is false
       const bool debug = false) {
      const FLOAT scale =
          is_expected_run ? 1.0f
          : expected_run_data->wealth_minus_npv_essential_at_start == 0.0
              ? 1.0
              : wealth_minus_npv_essential_at_start /
                    expected_run_data->wealth_minus_npv_essential_at_start;

      if (debug) {
        printf("\n----wealth_minus_npv_essential_at_start %.65f\n",
               wealth_minus_npv_essential_at_start);
        if (!is_expected_run) {
          printf("\n----expected_wealth_minus_npv_essential_at_start %.65f\n",
                 expected_run_data->wealth_minus_npv_essential_at_start);
        }
        printf("\n----withdrawal_started: %s\n",
               withdrawal_started ? "true" : "false");
        printf("\n----scale: %.65f\n", scale);
      }
      const CURRENCY general =
          !withdrawal_started
              ? 0.0
              : fmax((
                         // fma() is failing compilation with: "calling a
                         // constexpr __host__ function("fma") from a __device__
                         // function.
                         __fma_rn(-npv_approx_discretionary_with_current_month -
                                      npv_approx_legacy,
                                  scale,
                                  wealth_minus_npv_essential_at_start)) /
                         curr_month_cumulative_1_plus_g_over_1_plus_r,
                     0.0);
      return {
          .essential = current_month_essential_expense,
          .discretionary =
              fmax(current_month_discretionary_expense * scale, 0.0),
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
      }
    };

    template <bool write_result, bool is_expected_run>
    __device__ __forceinline__ _ReturnValue
    fn(const uint32_t run_index,
       const uint32_t month_index,
       const CURRENCY balance_starting,
       const uint32_t withdrawal_start_month,
       const OptCURRENCY spending_ceiling,
       const OptCURRENCY spending_floor,
       const CURRENCY current_month_income,
       const CURRENCY current_month_essential_expense,
       const CURRENCY current_month_discretionary_expense,
       const FLOAT current_month_stock_allocation_savings_portfolio,
       const StocksAndBondsFLOAT &returns,
       const Cuda_Processed_SPAW_Run_x_MFNSimulated_x_MFN::Entry
           &cuda_processed_run_x_mfn_simulated_x_mfn,
       _ExpectedRunData *expected_run_result_by_mfn,
       RunResultPadded_GPU *run_result_padded) {

      // const bool debug = run_index == 0 && month_index == 432;
      const bool debug = false;
      if (debug) {
        printf("\n----is_expected_run: %s\n",
               is_expected_run ? "true" : "false");
        printf("\n----write_result: %s\n", write_result ? "true" : "false");
        printf("\n----month_index: %u\n", month_index);
        printf("\n----run_index: %u\n", run_index);
        printf("\n----balance_starting: %.65f\n", balance_starting);
        printf("\n----cuda_processed_run_x_mfn_simulated_x_mfn: \n");
        cuda_processed_run_x_mfn_simulated_x_mfn.print(0);
      }

      // Step 1: Precomputation at start (inlined)
      const bool withdrawal_started = month_index >= withdrawal_start_month;
      const _PrecomputationAtStart precomputation_at_start =
          _get_precomputation_at_start::fn(
              balance_starting,
              current_month_income,
              current_month_essential_expense,
              current_month_discretionary_expense,
              cuda_processed_run_x_mfn_simulated_x_mfn.npv_approx
                  .income_bond_rate_without_current_month,
              cuda_processed_run_x_mfn_simulated_x_mfn.npv_approx
                  .income_portfolio_rate_without_current_month,
              cuda_processed_run_x_mfn_simulated_x_mfn.npv_approx
                  .essential_expenses_without_current_month,
              cuda_processed_run_x_mfn_simulated_x_mfn.npv_approx
                  .discretionary_expenses_without_current_month,
              debug);

      if (debug) {
        printf("\n----precomputation_at_start:\n");
        precomputation_at_start.print(0);
      }

      // Step 2: Data for expected run (if needed).
      if (is_expected_run && write_result) {
        expected_run_result_by_mfn[month_index] = {
            .wealth_minus_npv_essential_at_start =
                precomputation_at_start.wealth_minus_npv_essential_at_start};

        if (debug) {
          printf("\n----expected_run_result_by_mfn:\n");
          expected_run_result_by_mfn[month_index].print(0);
        }
      }

      // Step 3: Calculate target withdrawals before ceiling and floor (note
      // this  happens before contributions are applied).
      const run_common::TargetWithdrawals
          target_withdrawals_before_ceiling_and_floor =
              _get_target_withdrawals_assuming_no_ceiling_or_floor::fn<
                  is_expected_run>(
                  withdrawal_started,
                  cuda_processed_run_x_mfn_simulated_x_mfn
                      .cumulative_1_plus_g_over_1_plus_r,
                  precomputation_at_start.wealth_minus_npv_essential_at_start,
                  precomputation_at_start
                      .npv_approx_discretionary_expenses_with_current_month,
                  cuda_processed_run_x_mfn_simulated_x_mfn.npv_approx.legacy,
                  current_month_essential_expense,
                  current_month_discretionary_expense,
                  is_expected_run ? nullptr
                                  : &expected_run_result_by_mfn[month_index],
                  debug);

      if (debug) {
        printf("\n----target_withdrawals_before_ceiling_and_floor:\n");
        target_withdrawals_before_ceiling_and_floor.print(0);
      }

      // Step 4: Apply ceiling and floor to target withdrawals
      const run_common::TargetWithdrawals target_withdrawals =
          run_common::apply_withdrawal_ceiling_and_floor(
              target_withdrawals_before_ceiling_and_floor,
              // Note: this should NOT be the scaled discretionary expense.
              current_month_discretionary_expense,
              spending_ceiling,
              spending_floor,
              withdrawal_started,
              debug);

      if (debug) {
        printf("\n----target_withdrawals:\n");
        target_withdrawals.print(0);
      }

      // Step 5: Apply contributions and withdrawals
      const run_common::AfterContributionsAndWithdrawals
          savings_portfolio_after_contributions_and_withdrawals =
              run_common::apply_contributions_and_withdrawals(
                  balance_starting, current_month_income, target_withdrawals);

      if (debug) {
        printf(
            "\n----savings_portfolio_after_contributions_and_withdrawals:\n");
        savings_portfolio_after_contributions_and_withdrawals.print(0);
      }

      // Step 5.5: Handle NaN withdrawals from savings portfolio rate.
      const FLOAT withdrawals_from_savings_portfolio_rate =
          (!isfinite(savings_portfolio_after_contributions_and_withdrawals
                         .withdrawals.from_savings_portfolio_rate_or_nan_or_inf)
               ? ({
                   fn<false, is_expected_run>(
                       run_index,
                       month_index,
                       0.000001,
                       withdrawal_start_month,
                       spending_ceiling,
                       spending_floor,
                       current_month_income,
                       current_month_essential_expense,
                       current_month_discretionary_expense,
                       current_month_stock_allocation_savings_portfolio,
                       returns,
                       cuda_processed_run_x_mfn_simulated_x_mfn,
                       expected_run_result_by_mfn,
                       run_result_padded)
                       .withdrawals_from_savings_portfolio_rate;
                 })
               : savings_portfolio_after_contributions_and_withdrawals
                     .withdrawals.from_savings_portfolio_rate_or_nan_or_inf);

      // Step 6: Calculate stock allocation
      const FLOAT stock_allocation =
          current_month_stock_allocation_savings_portfolio;

      if (debug) {
        printf("\n----stock_allocation:\n");
        printf("    %.65f\n", stock_allocation);
      }

      // Step 7: Apply allocation
      const run_common::End savings_portfolio_at_end =
          run_common::apply_allocation(
              stock_allocation,
              returns,
              savings_portfolio_after_contributions_and_withdrawals.balance,
              cuda_processed_run_x_mfn_simulated_x_mfn.npv_approx
                  .income_bond_rate_without_current_month);

      if (debug) {
        printf("\n----savings_portfolio_at_end:\n");
        savings_portfolio_at_end.print(0);
      }

      if (debug) {
        printf("\n----important as array:\n");
        printf("%.1f,\n", static_cast<double>(run_index));
        printf("%.1f,\n", static_cast<double>(month_index));
        printf("%.65f,\n", balance_starting);
        printf("%.65f,\n",
               savings_portfolio_after_contributions_and_withdrawals
                   .contributions);
        printf("%.65f,\n",
               savings_portfolio_after_contributions_and_withdrawals.withdrawals
                   .essential);
        printf("%.65f,\n",
               savings_portfolio_after_contributions_and_withdrawals.withdrawals
                   .discretionary);
        printf("%.65f,\n",
               savings_portfolio_after_contributions_and_withdrawals.withdrawals
                   .general);
        printf("%.65f,\n",
               savings_portfolio_after_contributions_and_withdrawals.withdrawals
                   .total);
        printf("%.65f,\n", withdrawals_from_savings_portfolio_rate);
        printf("%.65f,\n",
               savings_portfolio_at_end
                   .stock_allocation_on_total_portfolio_or_zero_if_no_wealth);
        printf("%.65f,\n", stock_allocation);
        printf("%.65f,\n", savings_portfolio_at_end.balance);
      }

      // Step 8: Write results (in write_result)
      if (write_result && !is_expected_run) {
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
      };
    }
  } // namespace _single_month

  // *****************************************************************************
  // _kernel
  // *****************************************************************************
  template <bool is_expected_run>
  __global__ void
  _kernel(const uint32_t num_runs,
          const uint32_t num_months_to_simulate,
          const CURRENCY current_portfolio_balance,
          const uint32_t withdrawal_start_month,
          const OptCURRENCY spending_ceiling,
          const OptCURRENCY spending_floor,
          const StocksAndBondsFLOAT expected_returns_at_month_0,
          const CURRENCY *const income_by_mfn,
          const CURRENCY *const essential_expense_by_mfn,
          const CURRENCY *const discretionary_expense_by_mfn,
          const FLOAT *const stock_allocation_savings_portfolio_by_mfn,
          const StocksAndBondsFLOAT
              *const historical_returns_by_run_by_mfn_simulated,
          const Cuda_Processed_SPAW_Run_x_MFNSimulated_x_MFN::Entry
              *const cuda_processed_run_x_mfn_simulated_x_mfn,
          _ExpectedRunData *const expected_run_result_by_mfn,
          RunResultPadded_GPU *const run_result_padded) {

    const uint32_t run_index = (threadIdx.x + blockIdx.x * blockDim.x);
    if (run_index >= num_runs)
      return;

    CURRENCY balance_starting = current_portfolio_balance;
    for (uint32_t month_index = 0; month_index < num_months_to_simulate;
         month_index++) {
      const uint32_t run_by_mfn_simulated_index = get_run_by_mfn_index(
          num_runs, num_months_to_simulate, run_index, month_index);
      balance_starting =
          _single_month::fn<true, is_expected_run>(
              run_index,
              month_index,
              balance_starting,
              withdrawal_start_month,
              spending_ceiling,
              spending_floor,
              income_by_mfn[month_index],
              essential_expense_by_mfn[month_index],
              discretionary_expense_by_mfn[month_index],
              stock_allocation_savings_portfolio_by_mfn[month_index],
              is_expected_run ? expected_returns_at_month_0
                              : historical_returns_by_run_by_mfn_simulated
                                    [run_by_mfn_simulated_index],
              cuda_processed_run_x_mfn_simulated_x_mfn
                  [run_by_mfn_simulated_index],
              expected_run_result_by_mfn,
              run_result_padded)
              .balance_ending;
    }
  }

  // *****************************************************************************
  // run
  // *****************************************************************************

  RunResultPadded
  run(const uint32_t num_runs_in,
      const uint32_t num_months_to_simulate,
      const CURRENCY current_portfolio_balance,
      const uint32_t withdrawal_start_month,
      const OptCURRENCY spending_ceiling,
      const OptCURRENCY spending_floor,
      const StocksAndBondsFLOAT expected_returns_at_month_0,
      const thrust::device_vector<CURRENCY> &income_by_mfn,
      const thrust::device_vector<CURRENCY> &essential_expense_by_mfn,
      const thrust::device_vector<CURRENCY> &discretionary_expense_by_mfn,
      const thrust::device_vector<FLOAT>
          &stock_allocation_savings_portfolio_by_mfn,
      const thrust::device_vector<StocksAndBondsFLOAT>
          &historical_returns_by_run_by_mfn_simulated,
      const Cuda_Processed_SPAW_Run_x_MFNSimulated_x_MFN
          &cuda_processed_run_x_mfn_simulated_x_mfn) {

    thrust::device_vector<_ExpectedRunData> expected_run_result_by_mfn(
        num_months_to_simulate);
    RunResultPadded run_result_padded_host_struct =
        RunResultPadded::make(num_runs_in, num_months_to_simulate);
    unique_ptr_gpu<RunResultPadded_GPU> run_result_padded_device_struct =
        run_result_padded_host_struct.copy_to_gpu();

    const auto do_run = [&]<bool is_expected_run>() {
      const int32_t num_runs = is_expected_run ? 1 : num_runs_in;
      const int32_t block_size = 64;
      _kernel<is_expected_run>
          <<<(num_runs + block_size - 1) / block_size, block_size>>>(
              num_runs,
              num_months_to_simulate,
              current_portfolio_balance,
              withdrawal_start_month,
              spending_ceiling,
              spending_floor,
              expected_returns_at_month_0,
              income_by_mfn.data().get(),
              essential_expense_by_mfn.data().get(),
              discretionary_expense_by_mfn.data().get(),
              stock_allocation_savings_portfolio_by_mfn.data().get(),
              historical_returns_by_run_by_mfn_simulated.data().get(),
              is_expected_run
                  ? cuda_processed_run_x_mfn_simulated_x_mfn.for_expected_run
                        .data()
                        .get()
                  : cuda_processed_run_x_mfn_simulated_x_mfn.for_normal_run
                        .data()
                        .get(),
              expected_run_result_by_mfn.data().get(),
              run_result_padded_device_struct.get());
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());
    };

    do_run.operator()<true>();
    do_run.operator()<false>();

    return run_result_padded_host_struct;
  }

} // namespace spaw
