import { fGet, getLogReturns, getStats, getStatsWithLog } from '../Utils'

const _fromReturns = (returns: number[]) => {
  const stats = getStatsWithLog(returns)

  const convertExpectedToExpectedOfLog = (x: number) => {
    const naive = (x: number) => Math.log(1 + x) - stats.ofLog.variance / 2
    const delta = stats.ofLog.expectedValue - naive(stats.expectedValue)
    return naive(x) + delta
  }

  const adjust = (targetExpected: number) => {
    const targetExpectedOfLog = convertExpectedToExpectedOfLog(targetExpected)
    const adjustmentToLogReturns =
      stats.ofLog.expectedValue - targetExpectedOfLog
    const adjustedLogReturns = stats.ofLog.returns.map(
      (log) => log - adjustmentToLogReturns,
    )
    return adjustedLogReturns.map((x) => Math.exp(x) - 1)
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
