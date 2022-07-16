import {ReactNode, useMemo} from 'react'
import {createContext} from '../../Utils/CreateContext'
import {ChartPanelType} from '../Plan/ChartPanel/ChartPanelType'
import {
  tpawChartDataLegacy,
  TPAWChartDataLegacy,
} from '../Plan/ChartPanel/TPAWChart/TPAWChartDataLegacy'
import {
  TPAWChartDataMain,
  tpawChartDataMainPercentiles,
  tpawChartDataMainSharpeRatio,
} from '../Plan/ChartPanel/TPAWChart/TPAWChartDataMain'
import {useSimulation} from './WithSimulation'

type _Value = {
  byYearsFromNowPercentiles: Map<ChartPanelType, TPAWChartDataMain>
  sharpeRatio: TPAWChartDataMain | null
  legacy: TPAWChartDataLegacy
}

const [Context, useChartData] = createContext<_Value>('ChartData')

export {useChartData}

export const WithChartData = ({children}: {children: ReactNode}) => {
  const simulation = useSimulation()
  const {
    tpawResult,
    highlightPercentiles,
    percentiles: percentileList,
    forSharpeRatioComparison,
  } = simulation
  const value = useMemo(() => {
    const {params} = tpawResult.args
    const byYearsFromNowPercentiles = new Map<
      ChartPanelType,
      TPAWChartDataMain
    >()
    const _add = (type: ChartPanelType) => {
      byYearsFromNowPercentiles.set(
        type,
        tpawChartDataMainPercentiles(type, tpawResult, highlightPercentiles)
      )
    }

    _add('spending-total')
    _add('spending-general')
    params.withdrawals.essential.forEach(x => [
      x.id,
      _add(`spending-essential-${x.id}`),
    ])
    params.withdrawals.discretionary.forEach(x => [
      x.id,
      _add(`spending-discretionary-${x.id}`),
    ])
    _add('portfolio')
    _add('glide-path')
    _add('withdrawal')

    const sharpeRatio =
      forSharpeRatioComparison &&
      forSharpeRatioComparison.tpaw.tpawResult &&
      forSharpeRatioComparison.spaw.tpawResult &&
      forSharpeRatioComparison.swr.tpawResult
        ? tpawChartDataMainSharpeRatio(
            'sharpe-ratio',
            {
              tpaw: forSharpeRatioComparison.tpaw.tpawResult,
              spaw: forSharpeRatioComparison.spaw.tpawResult,
              swr: forSharpeRatioComparison.swr.tpawResult,
            },
            percentileList,
            highlightPercentiles
          )
        : null

    const legacy = tpawChartDataLegacy(tpawResult, highlightPercentiles)
    return {byYearsFromNowPercentiles, sharpeRatio, legacy}
  }, [
    forSharpeRatioComparison,
    highlightPercentiles,
    percentileList,
    tpawResult,
  ])

  return <Context.Provider value={value}>{children}</Context.Provider>
}
