#include "extern/doctest.h"
#include "mertons_formula.h"
#include "src/utils/cuda_utils.h"
#include <thrust/device_vector.h>

__host__ __device__ void
PlainMertonsFormulaClosure::print(uint32_t num_tabs) const {
  const uint32_t indent = num_tabs * 4;
  printf("%*sannual_equity_premium_by_variance: %.65f\n",
         indent,
         "",
         annual_equity_premium_by_variance);
  printf("%*sc0: %.65f\n", indent, "", c0);
  printf("%*sc1: %.65f\n", indent, "", c1);
  printf("%*sannual_additional_spending_tilt: %.65f\n",
         indent,
         "",
         annual_additional_spending_tilt);
}

__host__ __device__ void
EffectiveMertonsFormulaClosure::print(uint32_t num_tabs) const {
  const uint32_t indent = num_tabs * 4;
  printf("%*srra_for_all_stocks: %.65f\n", indent, "", rra_for_all_stocks);
  printf("%*splain_closure:\n", indent, "");
  plain_closure.print(num_tabs + 1);
}

__device__ __host__ void MertonsFormulaResult::print(uint32_t num_tabs) const {
  const uint32_t indent = num_tabs * 4;
  printf("%*sstock_allocation: %.65f\n", indent, "", stock_allocation);
  printf("%*sspending_tilt: %.65f\n", indent, "", spending_tilt);
}

// -----------------------------------------
// plain_mertons_formula
// -----------------------------------------
namespace _plain_mertons_formula {
  __global__ void _kernel(FLOAT annual_r,
                          FLOAT annual_equity_premium,
                          FLOAT annual_variance_stocks,
                          FLOAT rra_including_pos_infinity,
                          FLOAT time_preference,
                          FLOAT annual_additional_spending_tilt,
                          MertonsFormulaResult *results) {
    PlainMertonsFormulaClosure closure =
        get_plain_mertons_formula_closure(annual_r,
                                          annual_equity_premium,
                                          annual_variance_stocks,
                                          time_preference,
                                          annual_additional_spending_tilt);

    results[0] = plain_mertons_formula(closure, rra_including_pos_infinity);
  }

  MertonsFormulaResult _test_fn(FLOAT annual_r,
                                FLOAT annual_equity_premium,
                                FLOAT annual_variance_stocks,
                                FLOAT rra_including_pos_infinity,
                                FLOAT time_preference,
                                FLOAT annual_additional_spending_tilt) {
    thrust::device_vector<MertonsFormulaResult> results(1);
    _kernel<<<1, 1>>>(annual_r,
                      annual_equity_premium,
                      annual_variance_stocks,
                      rra_including_pos_infinity,
                      time_preference,
                      annual_additional_spending_tilt,
                      results.data().get());
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    return results[0];
  }

  TEST_CASE("STAR::plain_mertons_formula") {

    auto do_test = [&](const FLOAT annual_stock_returns,
                       const FLOAT annual_bond_returns,
                       const FLOAT annual_variance_stocks,
                       const FLOAT rra_including_pos_infinity,
                       const FLOAT time_preference,
                       const FLOAT annual_additional_spending_tilt,
                       const FLOAT truth_stock_allocation,
                       const FLOAT truth_spending_tilt) {
      thrust::device_vector<MertonsFormulaResult> d_results(1);

      const MertonsFormulaResult result =
          _test_fn(annual_bond_returns,
                   annual_stock_returns - annual_bond_returns,
                   annual_variance_stocks,
                   rra_including_pos_infinity,
                   time_preference,
                   annual_additional_spending_tilt);
      CHECK(result.stock_allocation == doctest::Approx(truth_stock_allocation));
      CHECK(result.spending_tilt == doctest::Approx(truth_spending_tilt));
    };

    const MertonsFormulaResult result =
        _test_fn(0.04, 0.0, 0.18 * 0.18, 3.04975986789893, .02, 0.0);
    result.print(0);

    SUBCASE("rra range") {
      SUBCASE("4") {
        do_test(0.05,                   // annual_stock_returns
                0.03,                   // annual_bond_returns
                0.01,                   // annual_variance_stocks
                4,                      // rra_including_pos_infinity
                0.01,                   // time_preference
                0.0,                    // annual_additional_spending_tilt
                0.5000000000000001,     // truth_stock_allocation
                0.0009327004773649339); // truth_spending_tilt
      }
      SUBCASE("0.04") {
        do_test(0.05,
                0.03,
                0.01,
                0.04,
                0.01,
                0.0,
                50.00000000000001,
                0.24962776727305425);
      }
      SUBCASE("Infinity") {
        do_test(0.05, 0.03, 0.01, INFINITY, 0.01, 0.0, 0.0, 0.0);
      }
    }
    SUBCASE("no equity premium") {
      do_test(0.03, 0.03, 0.01, 4.0, 0.01, 0.0, 0.0, 0.00041571484472902043);
    }
    SUBCASE("neg time preference") {
      do_test(0.05,
              0.03,
              0.01,
              4,
              -0.1,
              0.0,
              0.5000000000000001,
              0.003173196227570285);
    }
    SUBCASE("additional spending tilt") {
      do_test(0.05,
              0.03,
              0.01,
              4,
              0.01,
              1.0,
              0.5000000000000001,
              0.059958441910258564);
    }
    SUBCASE("max rra") {
      do_test(0.05,
              0.03,
              0.01,
              2.0000000000000004,
              0.01,
              0.0,
              1.0,
              0.002059836269842741);
    }
  };
} // namespace _plain_mertons_formula

// -----------------------------------------
// _get_rra_for_all_stocks
// -----------------------------------------

__global__ void test_get_rra_for_all_stocks(FLOAT annual_equity_premium,
                                            FLOAT annual_variance_stocks,
                                            FLOAT *result) {
  *result =
      get_rra_for_all_stocks(annual_equity_premium, annual_variance_stocks);
}

TEST_CASE("_get_rra_for_all_stocks") {
  auto do_test = [&](const FLOAT annual_stock_returns,
                     const FLOAT annual_bond_returns,
                     const FLOAT annual_variance_stocks,
                     const FLOAT truth) {
    thrust::device_vector<FLOAT> result_arr(1);
    test_get_rra_for_all_stocks<<<1, 1>>>(annual_stock_returns -
                                              annual_bond_returns,
                                          annual_variance_stocks,
                                          result_arr.data().get());
    CHECK(result_arr[0] == doctest::Approx(truth));
  };

  SUBCASE("typical case") { do_test(0.05, 0.03, 0.01, 2.0000000000000004); }
  SUBCASE("zero equity premium") { do_test(0.03, 0.03, 0.01, 0.0); }
}

// -----------------------------------------
// effective_mertons_formula
// -----------------------------------------
namespace _effective_mertons_formula {
  __global__ void _kernel(FLOAT annual_r,
                          FLOAT annual_equity_premium,
                          FLOAT annual_variance_stocks,
                          FLOAT rra_including_pos_infinity,
                          FLOAT time_preference,
                          FLOAT annual_additional_spending_tilt,
                          MertonsFormulaResult *results) {
    const EffectiveMertonsFormulaClosure closure =
        get_effective_mertons_formula_closure(annual_r,
                                              annual_equity_premium,
                                              annual_variance_stocks,
                                              time_preference,
                                              annual_additional_spending_tilt);
    results[0] = effective_mertons_formula(closure, rra_including_pos_infinity);
  }

  MertonsFormulaResult _test_fn(FLOAT annual_r,
                                FLOAT annual_equity_premium,
                                FLOAT annual_variance_stocks,
                                FLOAT rra_including_pos_infinity,
                                FLOAT time_preference,
                                FLOAT annual_additional_spending_tilt) {
    thrust::device_vector<MertonsFormulaResult> result_vec(1);
    _kernel<<<1, 1>>>(annual_r,
                      annual_equity_premium,
                      annual_variance_stocks,
                      rra_including_pos_infinity,
                      time_preference,
                      annual_additional_spending_tilt,
                      result_vec.data().get());
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    return result_vec[0];
  }

  TEST_CASE("effective_mertons_formula") {
    auto do_test = [&](const FLOAT annual_stock_returns,
                       const FLOAT annual_bond_returns,
                       const FLOAT sigma_pow2,
                       const FLOAT rra_including_pos_infinity,
                       const FLOAT time_preference,
                       const FLOAT annual_additional_spending_tilt,
                       const FLOAT truth_stock_allocation,
                       const FLOAT truth_spending_tilt) {
      thrust::device_vector<MertonsFormulaResult> d_results(1);

      const MertonsFormulaResult result =
          _test_fn(annual_bond_returns,
                   annual_stock_returns - annual_bond_returns,
                   sigma_pow2,
                   rra_including_pos_infinity,
                   time_preference,
                   annual_additional_spending_tilt);
      CHECK(result.stock_allocation == doctest::Approx(truth_stock_allocation));
      CHECK(result.spending_tilt == doctest::Approx(truth_spending_tilt));
    };

    SUBCASE("passthru") {
      do_test(0.05,
              0.03,
              0.01,
              4,
              0.01,
              0.0,
              0.5000000000000001,
              0.0009327004773649339);
    }
    SUBCASE("min rra") {
      do_test(0.05,
              0.03,
              0.01,
              2.0000000000000004,
              0.01,
              0.0,
              1.0,
              0.002059836269842741);
    }
    SUBCASE("< min rra") {
      do_test(0.05, 0.03, 0.01, 0.05, 0.01, 0.0, 1.0, 0.002059836269842741);
    }

    SUBCASE("neg equity premium") {
      do_test(0.02, 0.03, 0.01, 4.0, 0.01, 0.0, 0.0, 0.00041571484472902043);
    }
  }
} // namespace _effective_mertons_formula