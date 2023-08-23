import { useRouter } from 'next/router'
import { useCallback } from 'react'
import { Config } from '../../../Config'
import { PlanResultsChartType } from './PlanResultsChartType'

export function useGetPlanResultsChartURL() {
  const path = useRouter().asPath
  return useCallback(
    (type: PlanResultsChartType) => {
      return setPlanResultsChartURLOn(
        new URL(`${Config.client.urls.app()}${path}`),
        type,
      )
    },
    [path],
  )
}

export const setPlanResultsChartURLOn = (url: URL, type: PlanResultsChartType) => {
  if (type === 'spending-total') {
    url.searchParams.delete('graph')
  } else {
    url.searchParams.set('graph', type)
  }
  return url
}
