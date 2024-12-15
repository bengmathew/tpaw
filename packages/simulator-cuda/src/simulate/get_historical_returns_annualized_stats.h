#ifndef GET_HISTORICAL_RETURNS_STATS_H
#define GET_HISTORICAL_RETURNS_STATS_H

#include "src/public_headers/returns_stats.h"
#include "src/public_headers/stocks_and_bonds_float.h"
#include <thrust/device_vector.h>

ReturnsStats
get_historical_returns_stats(const uint32_t num_runs,
                             const uint32_t num_months_to_simulate,
                             const thrust::device_vector<StocksAndBondsFLOAT>
                                 &historical_returns_by_run_by_mfn_simulated);

#endif // GET_HISTORICAL_RETURNS_STATS_H
