#include "print_public_types.h"
#include "src/public_headers/stocks_and_bonds_float.h"
#include <cstdio>
#include <stdint.h>

__host__ __device__ void
print_stocks_and_bonds_float(const StocksAndBondsFLOAT &stocks_and_bonds,
                             const uint32_t num_tabs) {
  const uint32_t tabs = num_tabs * 4;
  printf("%*sstocks: %.65f\n", tabs, "", stocks_and_bonds.stocks);
  printf("%*sbonds: %.65f\n", tabs, "", stocks_and_bonds.bonds);
}

__host__ __device__ void
print_returns_stats_stats(const ReturnsStats::Stats &stats,
                          const uint32_t num_tabs) {
  const uint32_t tabs = num_tabs * 4;
  printf("%*sn: %u\n", tabs, "", stats.n);
  printf("%*smean: %.65f\n", tabs, "", stats.mean);
  printf("%*svariance: %.65f\n", tabs, "", stats.variance);
}

__host__ __device__ void print_returns_stats_log_and_non_log_stats(
    const ReturnsStats::LogAndNonLogStats &log_and_non_log_stats,
    const uint32_t num_tabs) {
  const uint32_t tabs = num_tabs * 4;
  printf("%*slog:\n", tabs, "");
  print_returns_stats_stats(log_and_non_log_stats.log, num_tabs + 1);
  printf("%*snon-log:\n", tabs, "");
  print_returns_stats_stats(log_and_non_log_stats.non_log, num_tabs + 1);
}

__host__ __device__ void print_returns_stats(const ReturnsStats &returns_stats,
                                             const uint32_t num_tabs) {
  const uint32_t tabs = num_tabs * 4;
  printf("%*sstocks:\n", tabs, "");
  print_returns_stats_log_and_non_log_stats(returns_stats.stocks, num_tabs + 1);
  printf("%*sbonds:\n", tabs, "");
  print_returns_stats_log_and_non_log_stats(returns_stats.bonds, num_tabs + 1);
}

__host__ __device__ void print_opt_currency(const OptCURRENCY &opt_currency,
                                            const uint32_t num_tabs) {
  const uint32_t tabs = num_tabs * 4;
  if (opt_currency.is_set) {
    printf("%*sopt_currency: Some(%.65f)\n", tabs, "", opt_currency.opt_value);
  } else {
    printf("%*sopt_currency: None\n", tabs, "");
  }
}
