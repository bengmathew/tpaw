import {GetStaticProps} from 'next'
import {getMarketData, MarketData} from '../Common/GetMarketData'

export const getMarketDataStaticProps: GetStaticProps<
  MarketData
> = async context => {
  const marketData = await getMarketData()
  return {props: marketData}
}
