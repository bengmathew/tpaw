import React from 'react'
import { getMarketDataStaticProps } from '../../src/Pages/App/GetMarketDataStaticProps'
import { WithChartData } from '../../src/Pages/App/WithChartData'
import { WithSimulation } from '../../src/Pages/App/WithSimulation'
import { MarketData } from '../../src/Pages/Common/GetMarketData'
import { Print } from '../../src/Pages/Print'

export default React.memo((marketData: MarketData) => (
  <WithSimulation marketData={marketData}>
    <WithChartData>
      <Print />
    </WithChartData>
  </WithSimulation>
))

export const getStaticProps = getMarketDataStaticProps
