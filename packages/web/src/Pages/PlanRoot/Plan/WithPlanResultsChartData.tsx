import _ from 'lodash'
import { ReactNode, useMemo } from 'react'
import { createContext } from '../../../Utils/CreateContext'
import { useSimulation } from '../PlanRootHelpers/WithSimulation'
import {
  PlanResultsChartData,
  getPlanResultsChartData,
} from './PlanResults/PlanResultsChartCard/PlanResultsChart/PlanResultsChartData'
import { PlanResultsChartType } from './PlanResults/PlanResultsChartType'
import { PERCENTILES_STR } from '../../../TPAWSimulator/Worker/TPAWRunInWorker'
import { PlanSizing } from './PlanSizing/PlanSizing'
import { PlanTransitionState } from './PlanTransition'
import { usePlanColors } from './UsePlanColors'
import { fGet } from '@tpaw/common'

type _Value = Map<PlanResultsChartType, PlanResultsChartData>

const [Context, useContext] = createContext<_Value>('ChartData')

export const useChartData = (type: PlanResultsChartType) =>
  fGet(useContext().get(type))

export const WithPlanResultsChartData = ({
  children,
  planSizing,
  planTransitionState,
}: {
  children: ReactNode
  planSizing: PlanSizing
  planTransitionState: PlanTransitionState
}) => {
  const simulation = useSimulation()
  const { tpawResult } = simulation
  const planColors = usePlanColors()
  const value = useMemo(() => {
    const { params } = tpawResult
    const result = new Map<PlanResultsChartType, PlanResultsChartData>()
    const _add = (type: PlanResultsChartType) => {
      result.set(
        type,
        getPlanResultsChartData(
          type,
          tpawResult,
          planSizing,
          planTransitionState,
          planColors,
        ),
      )
    }

    _add('spending-total')
    PERCENTILES_STR.forEach((percentile) =>
      _add(`spending-total-funding-sources-${percentile}`),
    )
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

    return result
  }, [planColors, planSizing, planTransitionState, tpawResult])

  return <Context.Provider value={value}>{children}</Context.Provider>
}
WithPlanResultsChartData.displayName = 'WithChartData'
