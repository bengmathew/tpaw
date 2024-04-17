import { MarketData } from '@tpaw/common'
import { GetStaticProps } from 'next'
import React from 'react'
import { getMarketData } from '../src/Pages/Common/GetMarketData'
import { ServerSidePrint } from '../src/Pages/ServerSidePrint/ServerSidePrint'

type _Props = {
  marketData: MarketData.Data
}

export default React.memo(({ marketData }: _Props) => (
  <ServerSidePrint marketData={marketData} />
))

export const getStaticProps: GetStaticProps<_Props> = async () => ({
  props: { marketData: await getMarketData() },
})
