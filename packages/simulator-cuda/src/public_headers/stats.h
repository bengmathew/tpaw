#ifndef STATS_H
#define STATS_H

#include "numeric_types.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct Stats {
  uint32_t n;
  FLOAT mean;
  FLOAT variance;
};

#ifdef __cplusplus
}
#endif

#endif // STATS_H
