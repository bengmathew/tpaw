import { MarketData, assert, fGet, getNYZonedTime, noCase } from '@tpaw/common'
import {
    JSONGuard,
    chain,
    constant,
    gte,
    integer,
    json,
    number,
    object,
    string,
    union,
} from 'json-guard'
import _ from 'lodash'
import { DateTime } from 'luxon'
import React, { ReactNode, useEffect, useState } from 'react'
import { createContext } from '../../../Utils/CreateContext'

const [Context, useMarketData] = createContext<{
  marketData: MarketData.Data
  synthesizeMarketDataSpec: SynthesizeMarketDataSpec | null
  setSynthesizeMarketDataSpec: React.Dispatch<
    React.SetStateAction<SynthesizeMarketDataSpec | null>
  >
  applySynthesizeMarketDataSpec: (() => void) | null
}>('MarketData')

export { useMarketData }


// TODO: Testing. Remove all market data from client side.
export const WithMarketData = React.memo(
  ({
    marketData: marketDataIn,
    children,
  }: {
    marketData: MarketData.Data
    children: ReactNode
  }) => {
    const [synthesizeMarketDataSpec, setSynthesizeMarketDataSpec] =
      useState<SynthesizeMarketDataSpec | null>(() =>
        _SynthesizeMarketData.read(),
      )

    const [startingSynthesizeMarketDataSpec] = useState(
      synthesizeMarketDataSpec,
    )
    useEffect(() => {
      _SynthesizeMarketData.write(synthesizeMarketDataSpec)
    }, [synthesizeMarketDataSpec])

    // Market data should stay const. Changes take effect only on reload.
    const [marketData] = useState(() => {
      const spec = startingSynthesizeMarketDataSpec
      if (!spec) return marketDataIn
      const currentTime = DateTime.now()
      return _SynthesizeMarketData.apply(
        marketDataIn,
        currentTime.minus({ years: spec.yearsBeforeNow }).toMillis(),
        currentTime.plus({ years: spec.yearsAfterNow }).toMillis(),
        spec.strategy,
      )
    })
    const applySynthesizeMarketDataSpec =
      startingSynthesizeMarketDataSpec === synthesizeMarketDataSpec
        ? null
        : () => window.location.reload()
    return (
      <Context.Provider
        value={{
          marketData,
          synthesizeMarketDataSpec,
          setSynthesizeMarketDataSpec,
          applySynthesizeMarketDataSpec,
        }}
      >
        {children}
      </Context.Provider>
    )
  },
)

export type SynthesizeMarketDataSpec = {
  v: 1
  yearsBeforeNow: number
  yearsAfterNow: number
  strategy: {
    dailyStockMarketPerformance:
      | {
          type: 'constant'
          annualVT: number
          annualBND: number
        }
      | { type: 'roundRobinOfRealData' }
      | { type: 'repeatGrowShrinkZero' }
  }
}

namespace _SynthesizeMarketData {
  const _guard: JSONGuard<SynthesizeMarketDataSpec> = object({
    v: constant(1),
    yearsBeforeNow: chain(number, integer, gte(0)),
    yearsAfterNow: chain(number, integer, gte(0)),
    strategy: object({
      dailyStockMarketPerformance: union(
        object({
          type: constant('constant'),
          annualVT: number,
          annualBND: number,
        }),
        object({ type: constant('roundRobinOfRealData') }),
        object({ type: constant('repeatGrowShrinkZero') }),
      ),
    }),
  })

  export const read = (): SynthesizeMarketDataSpec | null => {
    const src = window.localStorage.getItem(
      'WithMarketData_SynthesizeMarketDataSpec',
    )
    if (!src) return null
    const guardResult = chain(string, json, _guard)(src)
    if (guardResult.error) return null
    return guardResult.value
  }

  export const write = (spec: SynthesizeMarketDataSpec | null) => {
    if (!spec) {
      window.localStorage.removeItem('WithMarketData_SynthesizeMarketDataSpec')
      return
    }
    window.localStorage.setItem(
      'WithMarketData_SynthesizeMarketDataSpec',
      JSON.stringify(spec),
    )
  }

  const _splitMarketData = (marketData: MarketData.Data) => {
    const splitStream = <T extends { closingTime: number }>(
      getStream: (x: MarketData.Data[0]) => T,
    ) =>
      [
        ...new Map(
          marketData.map(getStream).map((x) => [x.closingTime, x]),
        ).values(),
      ].sort((a, b) => a.closingTime - b.closingTime)
    return {
      inflation: splitStream((x) => x.inflation),
      sp500: splitStream((x) => x.sp500),
      bondRates: splitStream((x) => x.bondRates),
      dailyStockMarketPerformance: splitStream(
        (x) => x.dailyStockMarketPerformance,
      ),
    }
  }
  export const apply = (
    marketDataIn: MarketData.Data,
    firstParamsTimestamp: number,
    estimationTimestamp: number,
    strategy: SynthesizeMarketDataSpec['strategy'],
  ): MarketData.Data => {
    const start = performance.now()
    const marketDataSplit = _splitMarketData(marketDataIn)
    let marketCloses = _synthesizeMarketCloseTimes(
      firstParamsTimestamp,
      estimationTimestamp,
    )
    const helper = <T extends { closingTime: number }>(
      original: T[],
      // Default is round robin of original values.
      getValue: (
        daysSinceLastMarketClose: number,
        index: number,
      ) => Omit<T, 'closingTime'> = (_, i) => original[i % original.length],
    ) => {
      return marketCloses.map((closingTime, i) => {
        const lastMarketClose =
          i === 0
            ? _marketCloseDelta(closingTime, -1).valueOf()
            : marketCloses[i - 1]
        const numDaysSinceLastMarketClose = Math.round(
          fGet(
            DateTime.fromMillis(marketCloses[i])
              .diff(DateTime.fromMillis(lastMarketClose), 'days')
              .toObject().days,
          ),
        )
        return { ...getValue(numDaysSinceLastMarketClose, i), closingTime }
      })
    }

    const inflation = helper(marketDataSplit.inflation)
    const sp500 = helper(marketDataSplit.sp500)
    const bondRates = helper(marketDataSplit.bondRates)

    const dailyStockMarketPerformance = helper(
      marketDataSplit.dailyStockMarketPerformance,
      strategy.dailyStockMarketPerformance.type === 'constant'
        ? (daysSinceLastMarketClose) => {
            assert(strategy.dailyStockMarketPerformance.type === 'constant')
            const fromAnnual = (annual: number) =>
              Math.pow(1 + annual, daysSinceLastMarketClose / 365) - 1
            return {
              percentageChangeFromLastClose: {
                vt: fromAnnual(strategy.dailyStockMarketPerformance.annualVT),
                bnd: fromAnnual(strategy.dailyStockMarketPerformance.annualBND),
              },
            }
          }
        : strategy.dailyStockMarketPerformance.type === 'repeatGrowShrinkZero'
        ? (_, i) => {
            const _fromGrow = (grow: number) =>
              [grow, 1 / (1 + grow) - 1, 0][i % 3]
            return {
              percentageChangeFromLastClose: {
                vt: _fromGrow(0.05),
                bnd: _fromGrow(0.05),
              },
            }
          }
        : strategy.dailyStockMarketPerformance.type === 'roundRobinOfRealData'
        ? undefined
        : noCase(strategy.dailyStockMarketPerformance),
    )

    const result = MarketData.combineStreams(
      inflation,
      sp500,
      bondRates,
      dailyStockMarketPerformance,
    )
    return result
  }

  let _marketCloses = [] as number[]
  const _synthesizeMarketCloseTimes = (startTime: number, endTime: number) => {
    const firstMarketClose = _marketCloseDelta(startTime, -1)
    if (_marketCloses.length === 0) _marketCloses.push(firstMarketClose)

    while (fGet(_.first(_marketCloses)) > firstMarketClose) {
      _marketCloses.unshift(
        _marketCloseDelta(fGet(_.first(_marketCloses)), -1).valueOf(),
      )
    }
    while (fGet(_.last(_marketCloses)) < endTime) {
      _marketCloses.push(
        _marketCloseDelta(fGet(_.last(_marketCloses)), 1).valueOf(),
      )
    }
    return _marketCloses.filter((x) => x >= firstMarketClose && x <= endTime)
  }

  const _marketCloseDelta = (
    currClosingTime: number,
    delta: -1 | 1,
  ): number => {
    const result = getNYZonedTime(currClosingTime)
      .plus({ day: delta })
      .set({ hour: 16, minute: 0, second: 0, millisecond: 0 })
    return result.weekdayShort === 'Sun' || result.weekdayShort === 'Sat'
      ? _marketCloseDelta(result.toMillis(), delta)
      : result.toMillis()
  }
}
