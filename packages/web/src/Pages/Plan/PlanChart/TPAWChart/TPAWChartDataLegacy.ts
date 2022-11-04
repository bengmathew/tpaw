import _ from 'lodash'
import {extendPlanParams} from '../../../../TPAWSimulator/PlanParamsExt'
import {SimpleRange} from '../../../../Utils/SimpleRange'
import {fGet} from '../../../../Utils/Utils'
import {SimulationInfo} from '../../../App/WithSimulation'

export type TPAWChartDataLegacy = {
  label: string
  percentiles: {
    data: number
    percentile: number
    isHighlighted: boolean
  }[]
  year: number
  xyDisplayRange: {x: SimpleRange; y: SimpleRange}
}

export function tpawChartDataLegacy(
  tpawResult: SimulationInfo['tpawResult'],
  highlightPercentiles: SimulationInfo['highlightPercentiles']
): TPAWChartDataLegacy {
  const {endingBalanceOfSavingsPortfolioByPercentile, args} = tpawResult
  const {numYears} = extendPlanParams(args.params.original)
  const percentiles = endingBalanceOfSavingsPortfolioByPercentile.map(x => ({
    data: x.data + args.params.legacy.tpawAndSPAW.external,
    percentile: x.percentile,
    isHighlighted: highlightPercentiles.includes(x.percentile),
  }))
  return {
    label: 'legacy',
    percentiles,
    year: numYears,
    xyDisplayRange: {
      x: {start: 0, end: 1},
      y: {
        start: 0,
        end: Math.max(0.0001, fGet(_.last(percentiles)).data),
      },
    },
  }
}
