import { fGet, letIn } from '../Utils'

export namespace MarketData {
  export type Inflation = {
    closingTime: number
    value: number
  }
  export type SP500 = {
    closingTime: number
    value: number
  }
  export type BondRates = {
    closingTime: number
    fiveYear: number
    sevenYear: number
    tenYear: number
    twentyYear: number
    thirtyYear: number
  }

  export type DailyStockMarketPerformance = {
    closingTime: number
    percentageChangeFromLastClose: {
      vt: number
      bnd: number
    }
  }

  export type Data = {
    closingTime: number
    inflation: Inflation
    sp500: SP500
    bondRates: BondRates
    dailyStockMarketPerformance: DailyStockMarketPerformance
  }[]

  export const combineStreams = (
    inflation: Inflation[],
    sp500: SP500[],
    bondRates: BondRates[],
    dailyStockMarketPerformance: DailyStockMarketPerformance[],
  ) => {
    const byClosingTimeWithMissing = new Map<
      number,
      {
        closingTime: number
        inflation: (typeof inflation)[0] | null
        sp500: (typeof sp500)[0] | null
        bondRates: (typeof bondRates)[0] | null
        dailyStockMarketPerformance:
          | (typeof dailyStockMarketPerformance)[0]
          | null
      }
    >()
    const set = (
      closingTime: number,
      setter: (
        value: Exclude<
          ReturnType<(typeof byClosingTimeWithMissing)['get']>,
          undefined
        >,
      ) => void,
    ) => {
      if (!byClosingTimeWithMissing.has(closingTime)) {
        byClosingTimeWithMissing.set(closingTime, {
          closingTime,
          inflation: null,
          sp500: null,
          bondRates: null,
          dailyStockMarketPerformance: null,
        })
      }
      setter(fGet(byClosingTimeWithMissing.get(closingTime)))
    }
    inflation.forEach((x) => set(x.closingTime, (y) => (y.inflation = x)))
    sp500.forEach((x) => set(x.closingTime, (y) => (y.sp500 = x)))
    bondRates.forEach((x) => set(x.closingTime, (y) => (y.bondRates = x)))
    dailyStockMarketPerformance.forEach((x) =>
      set(x.closingTime, (y) => (y.dailyStockMarketPerformance = x)),
    )
    const combinedNullable = letIn(
      [...byClosingTimeWithMissing.values()].sort(
        (a, b) => a.closingTime - b.closingTime,
      ),
      (x) =>
        x.slice(
          x.findIndex(
            (x) =>
              x.sp500 !== null &&
              x.inflation !== null &&
              x.bondRates !== null &&
              x.dailyStockMarketPerformance !== null,
          ),
        ),
    )

    const searchback = <T>(
      getter: (x: (typeof combinedNullable)[0]) => T | null,
      i: number,
    ): T => {
      const x: T | null = getter(fGet(combinedNullable[i]))
      if (x !== null) return x
      return searchback(getter, i - 1)
    }

    return combinedNullable.map((x, i) => ({
      closingTime: x.closingTime,
      inflation: searchback((x) => x.inflation, i),
      sp500: searchback((x) => x.sp500, i),
      bondRates: searchback((x) => x.bondRates, i),
      dailyStockMarketPerformance: searchback(
        (x) => x.dailyStockMarketPerformance,
        i,
      ),
    }))
  }
}
