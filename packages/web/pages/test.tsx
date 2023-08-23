import React from 'react'
import { getMarketDataStaticProps } from '../src/Pages/App/GetMarketDataStaticProps'
import { MarketData } from '@tpaw/common'
import { WithMarketData } from '../src/Pages/PlanRoot/PlanRootHelpers/WithMarketData'
import { Test } from '../src/Pages/Test'

export default React.memo((marketData: MarketData.Data) => (
  <WithMarketData marketData={marketData}>
    <Test />
  </WithMarketData>
))

export const getStaticProps = getMarketDataStaticProps
