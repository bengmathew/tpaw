import React from 'react'
import {getMarketDataStaticProps} from '../src/Pages/App/GetMarketDataStaticProps'
import {WithMarketData} from '../src/Pages/App/WithMarketData'
import {WithSimulation} from '../src/Pages/App/WithSimulation'
import {MarketData} from '../src/Pages/Common/GetMarketData'
import {TasksForThisYear} from '../src/Pages/TasksForThisYear/TasksForThisYear'

export default React.memo((marketData: MarketData) => (
  <WithMarketData marketData={marketData}>
    <WithSimulation>
      <TasksForThisYear />
    </WithSimulation>
  </WithMarketData>
))

export const getStaticProps = getMarketDataStaticProps
