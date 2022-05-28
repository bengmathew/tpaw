import _ from 'lodash'
import React, {ReactNode, useMemo} from 'react'
import {TPAWParams} from '../../TPAWSimulator/TPAWParams'
import {
  extendTPAWParams,
  TPAWParamsExt,
} from '../../TPAWSimulator/TPAWParamsExt'
import {
  processTPAWParams,
  TPAWParamsProcessed,
} from '../../TPAWSimulator/TPAWParamsProcessed'
import {
  useTPAWWorker,
  UseTPAWWorkerResult,
} from '../../TPAWSimulator/Worker/UseTPAWWorker'
import {createContext} from '../../Utils/CreateContext'
import {ChartPanelType} from '../Plan/ChartPanel/ChartPanelType'
import {
  TPAWChartDataMain,
  tpawChartDataMain,
} from '../Plan/ChartPanel/TPAWChart/TPAWChartDataMain'
import {useTPAWParams} from './UseTPAWParams'

export type SimulationInfo = {
  paramSpace: 'a' | 'b'
  setParamSpace: (space: 'a' | 'b') => void
  params: TPAWParams
  paramsProcessed: TPAWParamsProcessed
  paramsExt: TPAWParamsExt
  setParams: (params: TPAWParams | ((params: TPAWParams) => TPAWParams)) => void
  tpawResult: UseTPAWWorkerResult & {
    chartMainData: ReturnType<typeof _chartMainData>
  }
  numRuns: number
  percentiles: number[]
  highlightPercentiles: number[]
}
const [Context, useSimulation] = createContext<SimulationInfo>('TPAW')

const numRuns = 500
const highlightPercentiles = [5, 25, 50, 75, 95]
const percentiles = _.sortBy(_.union(_.range(5, 95, 2), highlightPercentiles))

export {useSimulation}

export const WithSimulation = ({children}: {children: ReactNode}) => {
  const {paramSpace, setParamSpace, params, setParams} = useTPAWParams()
  const paramsProcessed = useMemo(() => processTPAWParams(params), [params])
  const {resultInfo: tpawResult} = useTPAWWorker(
    paramsProcessed,
    numRuns,
    percentiles
  )
  const value = useMemo(
    () => ({
      paramSpace,
      setParamSpace,
      params,
      paramsProcessed: processTPAWParams(params),
      paramsExt: extendTPAWParams(params),
      setParams,
      numRuns,
      percentiles,
      highlightPercentiles,
      // Note, tpawResult will lag params. To get the exact params for the result, use the params object inside tpawResult.
      tpawResult: tpawResult
        ? {
            ...tpawResult,
            chartMainData: _chartMainData(tpawResult),
          }
        : null,
    }),
    [paramSpace, params, setParamSpace, setParams, tpawResult]
  )
  if (!_hasValue(value)) return <></>
  return <Context.Provider value={value}>{children}</Context.Provider>
}

const _hasValue = (x: {
  tpawResult: UseTPAWWorkerResult | null
}): x is {tpawResult: UseTPAWWorkerResult} => x.tpawResult !== null

function _chartMainData(
  tpawResult: UseTPAWWorkerResult
): Map<ChartPanelType, TPAWChartDataMain> {
  const {params} = tpawResult.args
  const result = new Map<ChartPanelType, TPAWChartDataMain>()
  const _add = (type: ChartPanelType) => {
    result.set(type, tpawChartDataMain(type, tpawResult, highlightPercentiles))
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
}
