#ifndef ANNUALIZE_SERIES_H
#define ANNUALIZE_SERIES_H

#include "src/public_headers/numeric_types.h"
#include <thrust/device_vector.h>

thrust::device_vector<FLOAT>
annualize_series(const thrust::device_vector<FLOAT> &series);

#endif // ANNUALIZE_SERIES_H