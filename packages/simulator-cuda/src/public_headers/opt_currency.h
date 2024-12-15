#ifndef OPT_CURRENCY_H
#define OPT_CURRENCY_H

#include "numeric_types.h"
#include <stdint.h> // bindgen cannot find cstdint (needed for sized number types).

#ifdef __cplusplus
extern "C" {
#endif

struct OptCURRENCY {
  uint32_t is_set;
  CURRENCY opt_value;
};

#ifdef __cplusplus
}
#endif

#endif // OPT_CURRENCY_H
