import { fGet } from '@tpaw/common'
import { Config } from '../Pages/Config'

export const sendAnalyticsEvent = (event: string, data: any) => {
  if (!Config.client.google.analytics.tagId) return
  fGet((window as any).gtag)('event', event, data)
}
