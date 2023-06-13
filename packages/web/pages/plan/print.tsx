import React from 'react'
import { getMarketDataStaticProps } from '../../src/Pages/App/GetMarketDataStaticProps'
import { WithChartData } from '../../src/Pages/App/WithChartData'
import { WithSimulation } from '../../src/Pages/App/WithSimulation'
import { WithWindowSize } from '../../src/Pages/App/WithWindowSize'
import { MarketData } from '../../src/Pages/Common/GetMarketData'
import { Print } from '../../src/Pages/Print/Print'

export default React.memo((marketData: MarketData) => (
  <WithWindowSize>
    <WithSimulation marketData={marketData}>
      <WithChartData>
        <Print />
      </WithChartData>
    </WithSimulation>
  </WithWindowSize>
))

export const getStaticProps = getMarketDataStaticProps
