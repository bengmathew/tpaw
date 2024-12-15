#include "extern/doctest.h"
#include "get_random_index.h"
#include "src/utils/cuda_utils.h"
#include <thrust/device_vector.h>
#include <vector>

// Thanks: https://stackoverflow.com/a/10989061/2771609
__device__ uint32_t get_random_index(uint32_t array_size,
                                     curandStateXORWOW_t *curand_state) {
  // Don't (possibly) narrow here to uint32_t, it might affect the
  // distribution.
  unsigned int x{curand(curand_state)};
  while (x >= UINT_MAX - (UINT_MAX % array_size)) {
    x = curand(curand_state);
  }
  // Ok to narrow here, array_size is uint32_t and this will be smaller.
  return static_cast<uint32_t>(x % array_size);
}

namespace test {
  __global__ void
  _kernal(const uint32_t array_size, const uint32_t n, uint32_t *out) {
    uint32_t run_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (run_index > 1)
      return;
    curandStateXORWOW_t curand_state;
    curand_init(1, 0, 0, &curand_state);
    for (uint32_t i = 0; i < n; ++i) {
      out[i] = get_random_index(array_size, &curand_state);
    }
  }

  std::vector<uint32_t> get_histogram(uint32_t array_size, uint32_t n) {
    thrust::device_vector<uint32_t> sample_device(n);

    _kernal<<<1, 32>>>(array_size, n, sample_device.data().get());
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    std::vector<uint32_t> sample_host = device_vector_to_host(sample_device);
    std::vector<uint32_t> hist(array_size, 0);
    for (auto i : sample_device) {
      hist[i]++;
    }
    return hist;
  }

  TEST_CASE("get_random_index") {
    const uint32_t array_size = 3;
    const uint32_t n = 100'000;
    auto hist = get_histogram(array_size, n);
    for (auto i : hist) {
      const double p = static_cast<double>(i) / static_cast<double>(n);
      CHECK(p - (1.0 / array_size) < 0.01);
    }
  }
} // namespace test
