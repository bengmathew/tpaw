import _ from 'lodash'
import {extendTPAWParams} from '../../../../TPAWSimulator/TPAWParamsExt'
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
}

export function tpawChartDataLegacy(
  tpawResult: SimulationInfo['tpawResult'],
  highlightPercentiles: SimulationInfo['highlightPercentiles']
): TPAWChartDataLegacy {
  const {endingBalanceOfSavingsPortfolioByPercentile, args} = tpawResult
  const {numYears} = extendTPAWParams(args.params.original)
  return {
    label: 'legacy',
    percentiles: endingBalanceOfSavingsPortfolioByPercentile.map(x => ({
      data: x.data + args.params.legacy.external,
      percentile: x.percentile,
      isHighlighted: highlightPercentiles.includes(x.percentile),
    })),
    year: numYears,
  }
}

export const tpawChartDataLegacyYRange = ({
  percentiles,
}: TPAWChartDataLegacy) => {
  return {
    start: 0,
    end: Math.max(0.0001, fGet(_.last(percentiles)).data),
  }
}
