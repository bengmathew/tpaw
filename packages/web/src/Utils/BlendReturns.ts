export const blendReturns =
  (returns: {stocks: number; bonds: number}) =>
  (allocation: {stocks: number}) =>
    returns.bonds * (1 - allocation.stocks) + returns.stocks * allocation.stocks
