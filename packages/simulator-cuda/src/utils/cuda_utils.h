#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda/std/utility>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <thrust/device_malloc_allocator.h>
#include <thrust/device_vector.h>

#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void
gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " "
              << line << std::endl;
    if (abort)
      exit(code);
  }
}

void *checkedCudaMalloc(size_t size);

template <typename T> using unique_ptr_gpu = std::unique_ptr<T, void (*)(T *)>;

template <typename T> unique_ptr_gpu<T> cudaMallocUnique() {
  void (*deleter)(T *) = [](T *ptr) { gpuErrchk(cudaFree(ptr)); };
  return unique_ptr_gpu<T>(static_cast<T *>(checkedCudaMalloc(sizeof(T))),
                           deleter);
}
template <typename T> unique_ptr_gpu<T> copyStructToDevice(const T &host) {
  unique_ptr_gpu<T> device = cudaMallocUnique<T>();
  cudaMemcpy(device.get(), &host, sizeof(T), cudaMemcpyHostToDevice);
  return device;
}

// This would not be needed if we used thrust::host_vector instead of
// std::vector (we could just use assignment), but we intentionally avoid
// thrust::host_vector because assignment is *too* easy. It makes it hard to
// track inadvertent copies from device to host.
template <typename T, typename A>
std::vector<T>
device_vector_to_host(const thrust::device_vector<T, A> &device) {
  std::vector<T> host(device.size());
  thrust::copy(device.begin(), device.end(), host.begin());
  return host;
}

inline double ms_since(clock_t start) {
  return static_cast<double>(clock() - start) / CLOCKS_PER_SEC * 1000;
}

#endif // CUDA_UTILS_H