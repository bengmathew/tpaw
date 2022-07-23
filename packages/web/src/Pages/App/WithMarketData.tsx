import React, {ReactElement} from 'react'
import {createContext} from '../../Utils/CreateContext'
import {MarketData} from '../Common/GetMarketData'

const [Context, useMarketData] = createContext<MarketData>('MarketData')

export {useMarketData}

export const WithMarketData = React.memo(
  ({
    children,
    marketData,
  }: {
    children: ReactElement
    marketData: MarketData
  }) => {
    return <Context.Provider value={marketData}>{children}</Context.Provider>
  }
)

