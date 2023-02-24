import React from 'react'
import {getMarketDataStaticProps} from '../../src/Pages/App/GetMarketDataStaticProps'
import {WithMarketData} from '../../src/Pages/App/WithMarketData'
import {WithSimulation} from '../../src/Pages/App/WithSimulation'
import {MarketData} from '../../src/Pages/Common/GetMarketData'
import {TasksForThisMonth} from '../../src/Pages/TasksForThisMonth/TasksForThisMonth'

export default React.memo((marketData: MarketData) => (
  <WithMarketData marketData={marketData}>
    <WithSimulation>
      <TasksForThisMonth />
    </WithSimulation>
  </WithMarketData>
))

export const getStaticProps = getMarketDataStaticProps
