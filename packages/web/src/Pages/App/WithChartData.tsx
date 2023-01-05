import { ReactNode, useMemo } from 'react'
import { createContext } from '../../Utils/CreateContext'
import { PlanChartType } from '../Plan/PlanChart/PlanChartType'
import {
  TPAWChartDataMain,
  tpawChartDataMainPercentiles,
} from '../Plan/PlanChart/TPAWChart/TPAWChartDataMain'
import { useSimulation } from './WithSimulation'

type _Value = {
  byYearsFromNowPercentiles: Map<PlanChartType, TPAWChartDataMain>
}

const [Context, useChartData] = createContext<_Value>('ChartData')

export { useChartData }

export const WithChartData = ({ children }: { children: ReactNode }) => {
  const simulation = useSimulation()
  const { tpawResult, highlightPercentiles } = simulation
  const value = useMemo(() => {
    const { params } = tpawResult.args
    const byYearsFromNowPercentiles = new Map<
      PlanChartType,
      TPAWChartDataMain
    >()
    const _add = (type: PlanChartType) => {
      byYearsFromNowPercentiles.set(
        type,
        tpawChartDataMainPercentiles(type, tpawResult, highlightPercentiles),
      )
    }

    _add('spending-total')
    _add('spending-general')
    params.original.adjustmentsToSpending.extraSpending.essential.forEach(
      (x) => [x.id, _add(`spending-essential-${x.id}`)],
    )
    params.original.adjustmentsToSpending.extraSpending.discretionary.forEach(
      (x) => [x.id, _add(`spending-discretionary-${x.id}`)],
    )
    _add('portfolio')
    _add('asset-allocation-savings-portfolio')
    _add('asset-allocation-total-portfolio')
    _add('withdrawal')

    return { byYearsFromNowPercentiles }
  }, [highlightPercentiles, tpawResult])

  return <Context.Provider value={value}>{children}</Context.Provider>
}
