import React from 'react'
import { getMarketDataStaticProps } from '../src/Pages/App/GetMarketDataStaticProps'
import { Perf } from '../src/Pages/Perf'
import { WithMarketData } from '../src/Pages/PlanRoot/PlanRootHelpers/WithMarketData'
import { MarketData } from '@tpaw/common'

export default React.memo(({ marketData }: { marketData: MarketData.Data }) => (
  <WithMarketData marketData={marketData}>
    <Perf />
  </WithMarketData>
))

export const getStaticProps = getMarketDataStaticProps
