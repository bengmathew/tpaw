#ifndef RETURNS_STATS_H
#define RETURNS_STATS_H

#include "numeric_types.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// TODO: Remove this in favor of Stats
struct ReturnsStats {
  struct Stats {
    uint32_t n;
    FLOAT mean;
    FLOAT variance;
  };

  struct LogAndNonLogStats {
    Stats log;
    Stats non_log;
  };

  LogAndNonLogStats stocks;
  LogAndNonLogStats bonds;
};

#ifdef __cplusplus
}
#endif

#endif // RETURNS_STATS_H
