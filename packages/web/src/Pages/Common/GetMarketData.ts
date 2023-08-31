import { MarketData, fGet } from '@tpaw/common'
import _ from 'lodash'
import { assert } from '../../Utils/Utils'
import { Config } from '../Config'

export const getMarketData = async (): Promise<MarketData.Data> => {
  const marketData = await fetch(Config.server.google.marketDataURL, {
    cache: 'no-store',
  })
  assert(marketData.ok)
  const result = (await marketData.json()) as MarketData.Data
  console.dir(fGet(_.last(result)).dailyStockMarketPerformance.closingTime)
  return result
}

export const getMarketDataIndexForTime = (
  timestamp: number,
  marketData: MarketData.Data,
) => {
  const index =
    _.sortedLastIndexBy<{ closingTime: number }>(
      marketData,
      { closingTime: timestamp },
      (x) => x.closingTime,
    ) - 1

  assert(index >= 0)
  return index
}

export const getMarketDataForTime = (
  timestamp: number,
  marketData: MarketData.Data,
) => marketData[getMarketDataIndexForTime(timestamp, marketData)]
