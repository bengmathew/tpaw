import {useRouter} from 'next/router'
import {useCallback} from 'react'
import {Config} from '../../Config'
import {PlanChartType} from './PlanChartType'

export function useGetPlanChartURL() {
  const path = useRouter().asPath
  return useCallback(
    (type: PlanChartType) => {
      const url = new URL(`${Config.client.urls.app()}${path}`)
      if (type === 'spending-total') {
        url.searchParams.delete('graph')
      } else {
        url.searchParams.set('graph', type)
      }
      return url
    },
    [path]
  )
}
