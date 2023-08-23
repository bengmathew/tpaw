import { MarketData } from '@tpaw/common'
import { GetStaticProps } from 'next'
import { getMarketData } from '../Common/GetMarketData'

export const getMarketDataStaticProps: GetStaticProps<{
  marketData: MarketData.Data
}> = async (context) => {
  const marketData = await getMarketData()
  return { props: { marketData } }
}
