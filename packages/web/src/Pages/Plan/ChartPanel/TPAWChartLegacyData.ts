import _ from 'lodash'
import {extendTPAWParams} from '../../../TPAWSimulator/TPAWParamsExt'
import {fGet} from '../../../Utils/Utils'
import {SimulationInfo} from '../../App/WithSimulation'

export type TPAWChartLegacyData = {
  label: string
  percentiles: {
    data: number
    percentile: number
    isHighlighted: boolean
  }[]
  year: number
}

export function tpawChartLegacyData(
  tpawResult: SimulationInfo['tpawResult'],
  highlightPercentiles: SimulationInfo['highlightPercentiles']
): TPAWChartLegacyData {
  const {legacyByPercentile, args} = tpawResult
  const {numYears} = extendTPAWParams(args.params)
  return {
    label: 'legacy',
    percentiles: legacyByPercentile.map(x => ({
      data: x.data,
      percentile: x.percentile,
      isHighlighted: highlightPercentiles.includes(x.percentile),
    })),
    year: numYears,
  }
}

export const tpawChartLegacyDataYRange = ({
  percentiles,
}: TPAWChartLegacyData) => {
  return {
    start: 0,
    end: Math.max(0.0001, fGet(_.last(percentiles)).data),
  }
}
