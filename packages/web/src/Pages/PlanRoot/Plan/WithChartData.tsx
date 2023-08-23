import { ReactNode, useEffect, useMemo } from 'react'
import { createContext } from '../../../Utils/CreateContext'
import { useSimulation } from '../PlanRootHelpers/WithSimulation'
import { PlanResultsChartType } from './PlanResults/PlanResultsChartType'
import {
  TPAWChartDataMain,
  tpawChartDataMain,
} from './PlanResults/TPAWChart/TPAWChartDataMain'
import _ from 'lodash'
import { fGet } from '@tpaw/common'

type _Value = {
  byYearsFromNowPercentiles: Map<PlanResultsChartType, TPAWChartDataMain>
}

const [Context, useChartData] = createContext<_Value>('ChartData')

export { useChartData }

export const WithChartData = ({ children }: { children: ReactNode }) => {
  const simulation = useSimulation()
  const { tpawResult } = simulation
  const value = useMemo(() => {
    const { params } = tpawResult
    const byYearsFromNowPercentiles = new Map<
      PlanResultsChartType,
      TPAWChartDataMain
    >()
    const _add = (type: PlanResultsChartType) => {
      byYearsFromNowPercentiles.set(type, tpawChartDataMain(type, tpawResult))
    }

    _add('spending-total')
    _add('spending-general')
    _.values(
      params.original.adjustmentsToSpending.extraSpending.essential,
    ).forEach((x) => [x.id, _add(`spending-essential-${x.id}`)])
    _.values(
      params.original.adjustmentsToSpending.extraSpending.discretionary,
    ).forEach((x) => [x.id, _add(`spending-discretionary-${x.id}`)])
    _add('portfolio')
    _add('asset-allocation-savings-portfolio')
    _add('asset-allocation-total-portfolio')
    _add('withdrawal')

    return { byYearsFromNowPercentiles }
  }, [tpawResult])

  

  return <Context.Provider value={value}>{children}</Context.Provider>
}
WithChartData.displayName = 'WithChartData'
