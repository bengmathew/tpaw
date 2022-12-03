import { ReactNode, useMemo } from 'react'
import { createContext } from '../../Utils/CreateContext'
import { PlanChartType } from '../Plan/PlanChart/PlanChartType'
import {
  TPAWChartDataMain,
  tpawChartDataMainPercentiles,
  tpawChartDataMainRewardRiskRatio,
} from '../Plan/PlanChart/TPAWChart/TPAWChartDataMain'
import { useSimulation } from './WithSimulation'

type _Value = {
  byYearsFromNowPercentiles: Map<
    Exclude<PlanChartType, 'reward-risk-ratio-comparison'>,
    TPAWChartDataMain
  >
  rewardRiskRatio: TPAWChartDataMain | null
}

const [Context, useChartData] = createContext<_Value>('ChartData')

export { useChartData }

export const WithChartData = ({ children }: { children: ReactNode }) => {
  const simulation = useSimulation()
  const {
    tpawResult,
    highlightPercentiles,
    percentiles: percentileList,
    forRewardRiskRatioComparison,
  } = simulation
  const value = useMemo(() => {
    const { params } = tpawResult.args
    const byYearsFromNowPercentiles = new Map<
      Exclude<PlanChartType, 'reward-risk-ratio-comparison'>,
      TPAWChartDataMain
    >()
    const _add = (
      type: Exclude<PlanChartType, 'reward-risk-ratio-comparison'>,
    ) => {
      byYearsFromNowPercentiles.set(
        type,
        tpawChartDataMainPercentiles(type, tpawResult, highlightPercentiles),
      )
    }

    _add('spending-total')
    _add('spending-general')
    params.original.extraSpending.essential.forEach((x) => [
      x.id,
      _add(`spending-essential-${x.id}`),
    ])
    params.original.extraSpending.discretionary.forEach((x) => [
      x.id,
      _add(`spending-discretionary-${x.id}`),
    ])
    _add('portfolio')
    _add('asset-allocation-savings-portfolio')
    _add('asset-allocation-total-portfolio')
    _add('withdrawal')

    const rewardRiskRatio =
      forRewardRiskRatioComparison &&
      forRewardRiskRatioComparison.tpaw.tpawResult &&
      forRewardRiskRatioComparison.spaw.tpawResult &&
      forRewardRiskRatioComparison.swr.tpawResult
        ? tpawChartDataMainRewardRiskRatio(
            'reward-risk-ratio-comparision',
            {
              tpaw: forRewardRiskRatioComparison.tpaw.tpawResult,
              spaw: forRewardRiskRatioComparison.spaw.tpawResult,
              swr: forRewardRiskRatioComparison.swr.tpawResult,
            },
            percentileList,
            highlightPercentiles,
          )
        : null

    return { byYearsFromNowPercentiles, rewardRiskRatio }
  }, [
    forRewardRiskRatioComparison,
    highlightPercentiles,
    percentileList,
    tpawResult,
  ])

  return <Context.Provider value={value}>{children}</Context.Provider>
}
