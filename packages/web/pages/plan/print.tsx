import React from 'react'
import { getMarketDataStaticProps } from '../../src/Pages/App/GetMarketDataStaticProps'
import { WithChartData } from '../../src/Pages/App/WithChartData'
import { WithMarketData } from '../../src/Pages/App/WithMarketData'
import { WithSimulation } from '../../src/Pages/App/WithSimulation'
import { MarketData } from '../../src/Pages/Common/GetMarketData'
import { Print } from '../../src/Pages/Print'

export default React.memo((marketData: MarketData) => (
  <WithMarketData marketData={marketData}>
    <WithSimulation>
      <WithChartData>
        <Print />
      </WithChartData>
    </WithSimulation>
  </WithMarketData>
))

export const getStaticProps = getMarketDataStaticProps
