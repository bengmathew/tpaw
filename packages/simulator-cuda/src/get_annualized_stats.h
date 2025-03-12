#ifndef GET_ANNUALIZED_STATS_H
#define GET_ANNUALIZED_STATS_H

#include "src/basic_types/log_and_non_log.h"
#include "src/basic_types/stocks_and_bonds.h"
#include "src/public_headers/numeric_types.h"
#include "src/public_headers/stats.h"
#include <thrust/device_vector.h>

LogAndNonLog<Stats> get_annualized_stats_from_indexes(
    const thrust::device_vector<uint32_t> &indexes_into_historical_returns,
    const uint32_t num_runs,
    const uint32_t num_months,
    const thrust::device_vector<FLOAT> &historical_monthly_log_returns);

StocksAndBonds<FLOAT> get_empirical_annual_non_log_mean(
    const uint64_t seed,
    const uint32_t num_runs,
    const uint32_t num_months,
    const thrust::device_vector<FLOAT> &historical_monthly_log_returns_stocks,
    const thrust::device_vector<FLOAT> &historical_monthly_log_returns_bonds);


#endif // GET_ANNUALIZED_STATS_H
