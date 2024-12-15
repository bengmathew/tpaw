#ifndef GET_RANDOM_INDEX_H
#define GET_RANDOM_INDEX_H

#include <cstdint>
#include <curand_kernel.h>

__device__ uint32_t get_random_index(uint32_t array_size,
                                     curandStateXORWOW_t *curand_state);

#endif // GET_RANDOM_INDEX_H