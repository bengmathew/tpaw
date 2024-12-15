#ifndef ACCOUNT_FOR_WITHDRAWAL_H
#define ACCOUNT_FOR_WITHDRAWAL_H

#include "src/public_headers/numeric_types.h"

struct AccountForWithdrawal {
  CURRENCY balance;
  bool insufficient_funds{false};

  __host__ __device__ __forceinline__ AccountForWithdrawal(CURRENCY balance)
      : balance(balance) {}

  __host__ __device__ __forceinline__ CURRENCY withdraw(CURRENCY amount) {
    if (balance < amount) {
      // We define insufficient funds as running out of at least $1. This is
      // prevent hypersensitivity when amount is pretty much equal to balance,
      // modula floating point imprecision.
      if (amount - balance > 1) {
        insufficient_funds = true;
      }
      const CURRENCY amount_withdrawn = balance;
      balance = 0;
      return amount_withdrawn;
    } else {
      balance -= amount;
      return amount;
    }
  }
};

#endif // ACCOUNT_FOR_WITHDRAWAL_H
