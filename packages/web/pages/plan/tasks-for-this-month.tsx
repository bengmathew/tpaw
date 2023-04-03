import React from 'react'
import { getMarketDataStaticProps } from '../../src/Pages/App/GetMarketDataStaticProps'
import { WithSimulation } from '../../src/Pages/App/WithSimulation'
import { MarketData } from '../../src/Pages/Common/GetMarketData'
import { TasksForThisMonth } from '../../src/Pages/TasksForThisMonth/TasksForThisMonth'

export default React.memo((marketData: MarketData) => (
  <WithSimulation marketData={marketData}>
    <TasksForThisMonth />
  </WithSimulation>
))

export const getStaticProps = getMarketDataStaticProps
