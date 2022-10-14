import { useURLParam } from '../../../Utils/UseURLParam'
import { useURLUpdater } from '../../../Utils/UseURLUpdater'
import { useChartData } from '../../App/WithChartData'
import { useSimulation } from '../../App/WithSimulation'
import { isPlanChartType } from './PlanChartType'
import { useGetPlanChartURL } from './UseGetPlanChartURL'

export function usePlanChartType() {
  const {tpawResult} = useSimulation()
  const chartData = useChartData()
  const urlUpdater = useURLUpdater()
  const getPlanChartURL = useGetPlanChartURL()

  const typeStr = useURLParam('graph') ?? ''
  let type = isPlanChartType(tpawResult.args.params.original, typeStr)
    ? typeStr
    : 'spending-total'
  if (
    type === 'reward-risk-ratio-comparison' &&
    chartData.rewardRiskRatio === null
  ) {
    type = 'spending-total'
  }
  if (typeStr.length > 0 && typeStr !== type)
    urlUpdater.replace(getPlanChartURL(type))

  return type
}
