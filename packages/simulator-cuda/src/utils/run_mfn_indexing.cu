#include "./run_mfn_indexing.h"
#include "extern/doctest.h"
#include "src/public_headers/numeric_types.h"
#include "src/utils/cuda_utils.h"

TEST_CASE("STAR::convert_to_run_major") {
  const uint32_t num_runs = 3;
  const uint32_t num_months = 5;
  std::vector<std::vector<FLOAT>> src{
      {1.0, 2.0, 3.0, 4.0, 5.0},
      {6.0, 7.0, 8.0, 9.0, 10.0},
      {11.0, 12.0, 13.0, 14.0, 15.0},
  };

  std::vector<FLOAT> value_by_run_by_mfn_host(
      static_cast<size_t>(num_runs * num_months));
  for (uint32_t run_index = 0; run_index < num_runs; ++run_index) {
    for (uint32_t month_index = 0; month_index < num_months; ++month_index) {
      value_by_run_by_mfn_host[get_run_by_mfn_index(
          num_runs, num_months, run_index, month_index)] =
          src[run_index][month_index];
    }
  }
  const thrust::device_vector<FLOAT> value_by_run_by_mfn =
      thrust::device_vector<FLOAT>(value_by_run_by_mfn_host);
  const std::vector<FLOAT> value_in_run_major = device_vector_to_host(
      convert_to_run_major(value_by_run_by_mfn, num_runs, num_months));

  for (uint32_t run_index = 0; run_index < num_runs; ++run_index) {
    for (uint32_t month_index = 0; month_index < num_months; ++month_index) {
      CHECK(src[run_index][month_index] ==
            value_in_run_major[RunMajor::get_run_by_mfn_index(
                num_months, run_index, month_index)]);
    }
  }
}

TEST_CASE("MonthMajor::get_run_by_mfn_index") {
  const uint32_t num_runs = 10;
  CHECK(MonthMajor::get_run_by_mfn_index(num_runs, 0, 0) == 0);
  CHECK(MonthMajor::get_run_by_mfn_index(num_runs, 1, 0) == 1);
  CHECK(MonthMajor::get_run_by_mfn_index(num_runs, 2, 0) == 2);
  CHECK(MonthMajor::get_run_by_mfn_index(num_runs, 9, 0) == 9);
  CHECK(MonthMajor::get_run_by_mfn_index(num_runs, 0, 1) == 10);
  CHECK(MonthMajor::get_run_by_mfn_index(num_runs, 1, 1) == 11);
}

TEST_CASE("MonthMajor::get_run_and_month_index") {
  const uint32_t num_runs = 10;
  CHECK(MonthMajor::get_run_and_month_index(num_runs, 0) ==
        RunAndMonthIndex{0, 0});
  CHECK(MonthMajor::get_run_and_month_index(num_runs, 1) ==
        RunAndMonthIndex{1, 0});
  CHECK(MonthMajor::get_run_and_month_index(num_runs, 2) ==
        RunAndMonthIndex{2, 0});
  CHECK(MonthMajor::get_run_and_month_index(num_runs, 9) ==
        RunAndMonthIndex{9, 0});
  CHECK(MonthMajor::get_run_and_month_index(num_runs, 10) ==
        RunAndMonthIndex{0, 1});
  CHECK(MonthMajor::get_run_and_month_index(num_runs, 11) ==
        RunAndMonthIndex{1, 1});
}

TEST_CASE("RunMajor::get_run_by_mfn_index") {
  const uint32_t num_months = 20;
  CHECK(RunMajor::get_run_by_mfn_index(num_months, 0, 0) == 0);
  CHECK(RunMajor::get_run_by_mfn_index(num_months, 0, 1) == 1);
  CHECK(RunMajor::get_run_by_mfn_index(num_months, 0, 2) == 2);
  CHECK(RunMajor::get_run_by_mfn_index(num_months, 0, 19) == 19);
  CHECK(RunMajor::get_run_by_mfn_index(num_months, 1, 0) == 20);
  CHECK(RunMajor::get_run_by_mfn_index(num_months, 1, 1) == 21);
}

TEST_CASE("RunMajor::get_run_and_month_index") {
  const uint32_t num_months = 20;
  CHECK(RunMajor::get_run_and_month_index(num_months, 0) ==
        RunAndMonthIndex{0, 0});
  CHECK(RunMajor::get_run_and_month_index(num_months, 1) ==
        RunAndMonthIndex{0, 1});
  CHECK(RunMajor::get_run_and_month_index(num_months, 2) ==
        RunAndMonthIndex{0, 2});
  CHECK(RunMajor::get_run_and_month_index(num_months, 19) ==
        RunAndMonthIndex{0, 19});
  CHECK(RunMajor::get_run_and_month_index(num_months, 20) ==
        RunAndMonthIndex{1, 0});
  CHECK(RunMajor::get_run_and_month_index(num_months, 21) ==
        RunAndMonthIndex{1, 1});
}

TEST_CASE("get_run_by_mfn_index_from_run_major_index") {
  const uint32_t num_runs = 10;
  const uint32_t num_months = 20;

  CHECK(MonthMajor::get_run_by_mfn_index_from_run_major_index(
            num_runs, num_months, 0) == 0);
  CHECK(MonthMajor::get_run_by_mfn_index_from_run_major_index(
            num_runs, num_months, 1) == 10);
  CHECK(MonthMajor::get_run_by_mfn_index_from_run_major_index(
            num_runs, num_months, 2) == 20);
  CHECK(MonthMajor::get_run_by_mfn_index_from_run_major_index(
            num_runs, num_months, 19) == 190);
  CHECK(MonthMajor::get_run_by_mfn_index_from_run_major_index(
            num_runs, num_months, 20) == 1);
  CHECK(MonthMajor::get_run_by_mfn_index_from_run_major_index(
            num_runs, num_months, 21) == 11);
}
