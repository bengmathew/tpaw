import { MarketData, block } from '@tpaw/common'
import _ from 'lodash'
import { assert } from '../../Utils/Utils'
import { Config } from '../Config'

export const getMarketData = async (): Promise<MarketData.Data> => {
  const url = await block(async () => {
    const authHeaders = _.compact([Config.client.debug.authHeader])

    const response = await fetch(
      `${Config.client.urls.backend}/marketDataURL`,
      {
        cache: 'no-store',
        headers: {
          ...(authHeaders ? { Authorization: authHeaders.join(', ') } : {}),
        },
      },
    )
    return await response.text()
  })
  const marketData = await fetch(url, {
    cache: 'no-store',
  })
  assert(marketData.ok)
  const result = (await marketData.json()) as MarketData.Data
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
