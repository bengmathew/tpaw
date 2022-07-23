import React from 'react'
import {WithChartData} from '../src/Pages/App/WithChartData'
import {WithMarketData} from '../src/Pages/App/WithMarketData'
import {WithSimulation} from '../src/Pages/App/WithSimulation'
import {WithWindowSize} from '../src/Pages/App/WithWindowSize'
import {Plan} from '../src/Pages/Plan/Plan'
import {
  planGetStaticProps,
  PlanStaticProps,
} from '../src/Pages/Plan/PlanGetStaticProps'

export default React.memo(({content, marketData}: PlanStaticProps) => (
  <WithWindowSize>
    <WithMarketData marketData={marketData}>
      <WithSimulation>
        <WithChartData>
          <Plan {...content} />
        </WithChartData>
      </WithSimulation>
    </WithMarketData>
  </WithWindowSize>
))

export const getStaticProps = planGetStaticProps
