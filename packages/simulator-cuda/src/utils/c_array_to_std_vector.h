
#ifndef C_ARRAY_TO_STD_VECTOR_H
#define C_ARRAY_TO_STD_VECTOR_H

#include <cstdint>
#include <vector>

template <typename T>
std::vector<T> c_array_to_std_vector(const T *const ptr, const uint32_t len) {
  return std::vector<T>(ptr, ptr + len);
}

#endif // C_ARRAY_TO_STD_VECTOR_H
