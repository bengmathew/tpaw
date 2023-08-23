import { Contentful } from '../../../../../../Utils/Contentful'
import { formatPercentage } from '../../../../../../Utils/FormatPercentage'
import { useSimulation } from '../../../../PlanRootHelpers/WithSimulation'
import { PlanInputType } from '../../Helpers/PlanInputType'
import { useMarketData } from '../../../../PlanRootHelpers/WithMarketData'
import { useNonPlanParams } from '../../../../PlanRootHelpers/WithNonPlanParams'
import { usePlanContent } from '../../../../PlanRootHelpers/WithPlanContent'
import { useMemo } from 'react'
import { getMarketDataForTime } from '../../../../../Common/GetMarketData'

export function usePlanInputGuideContent(type: PlanInputType) {
  const { marketData } = useMarketData()
  const { nonPlanParams } = useNonPlanParams()
  const { planParams, currentMarketData } = useSimulation()
  const { CAPE, bondRates, inflation } = currentMarketData

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
    capeDate: formatDate(CAPE.closingTime),
    expectedReturnsStocksCAPE: CAPE.value.toFixed(2),
    expectedReturnsStocksOneOverCAPE: formatPercentage(1)(CAPE.oneOverCAPE),
    expectedReturnsStocksRegressionFull5Year: formatPercentage(1)(
      CAPE.regression.full.fiveYear,
    ),
    expectedReturnsStocksRegressionFull10Year: formatPercentage(1)(
      CAPE.regression.full.tenYear,
    ),
    expectedReturnsStocksRegressionFull20Year: formatPercentage(1)(
      CAPE.regression.full.twentyYear,
    ),
    expectedReturnsStocksRegressionFull30Year: formatPercentage(1)(
      CAPE.regression.full.thirtyYear,
    ),
    expectedReturnsStocksRegressionRestricted5Year: formatPercentage(1)(
      CAPE.regression.restricted.fiveYear,
    ),
    expectedReturnsStocksRegressionRestricted10Year: formatPercentage(1)(
      CAPE.regression.restricted.tenYear,
    ),
    expectedReturnsStocksRegressionRestricted20Year: formatPercentage(1)(
      CAPE.regression.restricted.twentyYear,
    ),
    expectedReturnsStocksRegressionRestricted30Year: formatPercentage(1)(
      CAPE.regression.restricted.thirtyYear,
    ),
    expectedReturnsRegresssionAverage: formatPercentage(1)(
      CAPE.regressionAverage,
    ),
    expectedReturnsSuggested: formatPercentage(1)(CAPE.suggested),
    bondsDate: formatDate(bondRates.closingTime),
    expectedReturnsBonds5Year: formatPercentage(1)(bondRates.fiveYear),
    expectedReturnsBonds10Year: formatPercentage(1)(bondRates.tenYear),
    expectedReturnsBonds20Year: formatPercentage(1)(bondRates.twentyYear),
    expectedReturnsBonds30Year: formatPercentage(1)(bondRates.thirtyYear),
    inflationDate: formatDate(inflation.closingTime),
    inflation: formatPercentage(1)(inflation.value),
  }
  return Contentful.replaceVariables(variables, content)
}
