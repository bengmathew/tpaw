import { CallRust } from '../../../../../../UseSimulator/PlanParamsProcessed/CallRust'
import { fWASM } from '../../../../../../UseSimulator/Simulator/GetWASM'
import { Contentful } from '../../../../../../Utils/Contentful'
import { formatPercentage } from '../../../../../../Utils/FormatPercentage'
import { useNonPlanParams } from '../../../../PlanRootHelpers/WithNonPlanParams'
import { usePlanContent } from '../../../../PlanRootHelpers/WithPlanContent'
import { useSimulation } from '../../../../PlanRootHelpers/WithSimulation'
import { PlanInputType } from '../../Helpers/PlanInputType'

export function usePlanInputGuideContent(type: PlanInputType) {
  const { nonPlanParams } = useNonPlanParams()
  const { planParams, defaultPlanParams, currentMarketData } = useSimulation()
  const { inflation } = currentMarketData

  const formatDate = (epoch: number) =>
    new Date(epoch).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      timeZone: 'UTC',
    })

  const presetInfo =
    fWASM().process_market_data_for_expected_returns_for_planning_presets(
      planParams.advanced.sampling,
      planParams.advanced.historicalMonthlyLogReturnsAdjustment
        .standardDeviation,
      currentMarketData,
    )

  const contentForType = usePlanContent()[type]
  if (!('guide' in contentForType)) return null
  const content = contentForType.guide[planParams.advanced.strategy]
  const variables = {
    numRuns: `${nonPlanParams.numOfSimulationForMonteCarloSampling}`,
    inflationDate: formatDate(inflation.closingTime),
    inflation: formatPercentage(1)(inflation.value),
    historicalExpectedStockReturn: formatPercentage(1)(
      presetInfo.stocks.historical,
    ),
    historicalExpectedBondReturn: formatPercentage(1)(
      presetInfo.stocks.historical,
    ),
  }
  return Contentful.replaceVariables(variables, content)
}
