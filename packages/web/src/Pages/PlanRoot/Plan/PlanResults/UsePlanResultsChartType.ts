import { useURLParam } from '../../../../Utils/UseURLParam'
import { useURLUpdater } from '../../../../Utils/UseURLUpdater'
import { useSimulation } from '../../PlanRootHelpers/WithSimulation'
import { isPlanResultsChartType } from './PlanResultsChartType'
import { useGetPlanResultsChartURL } from './UseGetPlanResultsChartURL'

export function usePlanResultsChartType() {
  const { simulationResult } = useSimulation()
  const urlUpdater = useURLUpdater()
  const getPlanResultsChartURL = useGetPlanResultsChartURL()

  const typeStr = useURLParam('graph') ?? ''
  let type = isPlanResultsChartType(
    simulationResult.args.planParamsNorm,
    typeStr,
  )
    ? typeStr
    : 'spending-total'
  if (typeStr.length > 0 && typeStr !== type)
    urlUpdater.replace(getPlanResultsChartURL(type))

  return type
}
