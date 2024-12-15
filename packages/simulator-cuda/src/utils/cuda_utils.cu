#include "cuda_utils.h"
#include "extern/doctest.h"
#include <thrust/device_vector.h>

void *checkedCudaMalloc(size_t size) {
  void *ptr{nullptr};
  gpuErrchk(cudaMalloc(&ptr, size));
  return ptr;
}
