#include "extern/doctest.h"
#include "src/public_headers/numeric_types.h"
#include <stdio.h>

__global__ void test_kernel() {}

TEST_CASE("cuda_test") {
  test_kernel<<<1, 1>>>();
  cudaDeviceSynchronize();
};