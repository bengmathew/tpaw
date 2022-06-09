import {ReactNode, useMemo} from 'react'
import {createContext} from '../../Utils/CreateContext'
import {ChartPanelType} from '../Plan/ChartPanel/ChartPanelType'
import {
  TPAWChartDataMain,
  tpawChartDataMain,
} from '../Plan/ChartPanel/TPAWChart/TPAWChartDataMain'
import {useSimulation} from './WithSimulation'

type _Value = Map<ChartPanelType, TPAWChartDataMain>

const [Context, useChartMainData] = createContext<_Value>('ChartMainData')

export {useChartMainData}

export const WithChartMainData = ({children}: {children: ReactNode}) => {
  const {tpawResult, highlightPercentiles} = useSimulation()
  const value = useMemo(() => {
    const {params} = tpawResult.args
    const result = new Map<ChartPanelType, TPAWChartDataMain>()
    const _add = (type: ChartPanelType) => {
      result.set(
        type,
        tpawChartDataMain(type, tpawResult, highlightPercentiles)
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
    _add('withdrawal-rate')

    return result
  }, [tpawResult, highlightPercentiles])

  return <Context.Provider value={value}>{children}</Context.Provider>
}
