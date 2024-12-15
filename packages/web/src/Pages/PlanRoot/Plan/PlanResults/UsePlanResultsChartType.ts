import { useURLParam } from '../../../../Utils/UseURLParam'
import { useURLUpdater } from '../../../../Utils/UseURLUpdater'
import { useSimulationResultInfo } from '../../PlanRootHelpers/WithSimulation'
import { getPlanResultsChartTypeFromStr } from './PlanResultsChartType'
import { useGetPlanResultsChartURL } from './UseGetPlanResultsChartURL'

export function usePlanResultsChartType() {
  const { simulationResult } = useSimulationResultInfo()
  const urlUpdater = useURLUpdater()
  const getPlanResultsChartURL = useGetPlanResultsChartURL()

  const typeStr = useURLParam('graph') ?? ''
  let type =
    getPlanResultsChartTypeFromStr(
      simulationResult.planParamsNormOfResult,
      typeStr,
    ) ?? 'spending-total'
  if (typeStr.length > 0 && typeStr !== type)
    urlUpdater.replace(getPlanResultsChartURL(type))

  return type
}
