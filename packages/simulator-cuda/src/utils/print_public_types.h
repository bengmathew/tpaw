#ifndef PRINT_PUBLIC_TYPES_H
#define PRINT_PUBLIC_TYPES_H

#include "src/public_headers/returns_stats.h"
#include "src/public_headers/stocks_and_bonds_float.h"
#include "src/public_headers/opt_currency.h"
#include <stdint.h>

__host__ __device__ void
print_stocks_and_bonds_float(const StocksAndBondsFLOAT &stocks_and_bonds,
                             const uint32_t num_tabs);

__host__ __device__ void
print_returns_stats_stats(const ReturnsStats::Stats &stats,
                          const uint32_t num_tabs);

__host__ __device__ void print_returns_stats_log_and_non_log_stats(
    const ReturnsStats::LogAndNonLogStats &log_and_non_log_stats,
    const uint32_t num_tabs);

__host__ __device__ void print_returns_stats(const ReturnsStats &returns_stats,
                                             const uint32_t num_tabs);

__host__ __device__ void print_opt_currency(const OptCURRENCY &opt_currency,
                                             const uint32_t num_tabs);

#endif // PRINT_PUBLIC_TYPES_H
