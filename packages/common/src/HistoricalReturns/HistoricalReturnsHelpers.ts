import { fGet, getStatsWithLog } from '../Utils'

const _convertExpectedToExpectedOfLog = (
  targetExpected: number,
  stats: ReturnType<typeof getStatsWithLog>,
) => {
  const naive = (x: number) => Math.log(1 + x) - stats.ofLog.variance / 2
  const delta = stats.ofLog.expectedValue - naive(stats.ofBase.expectedValue)
  return naive(targetExpected) + delta
}

const _adjustExpected = (
  targetExpected: number,
  stats: ReturnType<typeof getStatsWithLog>,
) => {
  const targetExpectedOfLog = _convertExpectedToExpectedOfLog(
    targetExpected,
    stats,
  )
  const adjustmentToLogReturns = stats.ofLog.expectedValue - targetExpectedOfLog
  const adjustedLogReturns = stats.ofLog.returns.map(
    (log) => log - adjustmentToLogReturns,
  )
  return adjustedLogReturns.map((x) => Math.exp(x) - 1)
}

const _fromReturns = (returns: number[]) => {
  const stats = getStatsWithLog(returns)

  const adjust = (targetExpected: number, volatilityScale: number) => {
    const volatilityAdjustedStats =
      // Optimization for the common case where volatilityScale is 1.
      volatilityScale === 1
        ? stats
        : getStatsWithLog(
            stats.ofLog.returns.map((log) => {
              return (
                Math.exp(
                  stats.ofLog.expectedValue +
                    (log - stats.ofLog.expectedValue) * volatilityScale,
                ) - 1
              )
            }),
          )
    return _adjustExpected(targetExpected, volatilityAdjustedStats)
  }

  return {
    ...stats,
    adjust,
  }
}

export const returnsFromRawInput = (
  csvOrRaw: string | { stocks: number; bonds: number }[],
) => {
  const raw =
    typeof csvOrRaw === 'string'
      ? csvOrRaw
          .split('\n')
          .map((x) =>
            x
              .trim()
              .split(',')
              .map((x) => x.trim()),
          )
          .map(([period, stocks, bonds]) => ({
            period: fGet(period),
            stocks: parseFloat(fGet(stocks)),
            bonds: parseFloat(fGet(bonds)),
          }))
      : csvOrRaw

  return {
    stocks: _fromReturns(raw.map((x) => x.stocks)),
    bonds: _fromReturns(raw.map((x) => x.bonds)),
  }
}
