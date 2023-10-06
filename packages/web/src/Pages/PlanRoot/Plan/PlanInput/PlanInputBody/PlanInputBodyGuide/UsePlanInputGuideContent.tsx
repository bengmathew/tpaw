import { historicalReturns } from '@tpaw/common'
import { Contentful } from '../../../../../../Utils/Contentful'
import { formatCurrency } from '../../../../../../Utils/FormatCurrency'
import { formatPercentage } from '../../../../../../Utils/FormatPercentage'
import {
  useIANATimezoneName,
  useNonPlanParams,
} from '../../../../PlanRootHelpers/WithNonPlanParams'
import { usePlanContent } from '../../../../PlanRootHelpers/WithPlanContent'
import { useSimulation } from '../../../../PlanRootHelpers/WithSimulation'
import { PlanInputType } from '../../Helpers/PlanInputType'

export function usePlanInputGuideContent(type: PlanInputType) {
  const { getZonedTime } = useIANATimezoneName()
  const { nonPlanParams } = useNonPlanParams()
  const { planParams, currentMarketData } = useSimulation()
  const {  inflation } = currentMarketData

  const formatDate = (epoch: number) =>
    new Date(epoch).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      timeZone: 'UTC',
    })

  const contentForType = usePlanContent()[type]
  if (!('guide' in contentForType)) return null
  const content = contentForType.guide[planParams.advanced.strategy]
  const variables = {
    numRuns: `${nonPlanParams.numOfSimulationForMonteCarloSampling}`,
    inflationDate: formatDate(inflation.closingTime),
    inflation: formatPercentage(1)(inflation.value),
    historicalExpectedStockReturn: formatPercentage(1)(
      historicalReturns.monthly.annualStats.stocks.ofBase.mean,
    ),
    historicalExpectedBondReturn: formatPercentage(1)(
      historicalReturns.monthly.annualStats.bonds.ofBase.mean,
    ),
  }
  return Contentful.replaceVariables(variables, content)
}
