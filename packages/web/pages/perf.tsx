import React from 'react'
import {getMarketDataStaticProps} from '../src/Pages/App/GetMarketDataStaticProps'
import {WithMarketData} from '../src/Pages/App/WithMarketData'
import {MarketData} from '../src/Pages/Common/GetMarketData'
import {Perf} from '../src/Pages/Perf'

export default React.memo((marketData: MarketData) => (
  <WithMarketData marketData={marketData}>
    <Perf />
  </WithMarketData>
))

export const getStaticProps = getMarketDataStaticProps
