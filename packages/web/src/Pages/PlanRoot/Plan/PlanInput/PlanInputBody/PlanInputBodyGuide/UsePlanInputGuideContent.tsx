import { CalendarMonthFns } from '@tpaw/common'
import { Contentful } from '../../../../../../Utils/Contentful'
import { formatPercentage } from '../../../../../../Utils/FormatPercentage'
import { useNonPlanParams } from '../../../../PlanRootHelpers/WithNonPlanParams'
import { usePlanContent } from '../../../../PlanRootHelpers/WithPlanContent'
import { useSimulation } from '../../../../PlanRootHelpers/WithSimulation'
import { PlanInputType } from '../../Helpers/PlanInputType'

export function usePlanInputGuideContent(type: PlanInputType) {
  const { nonPlanParams } = useNonPlanParams()
  const { planParamsNorm, simulationResult } = useSimulation()
  const { marketData } = simulationResult.info

  const formatDate = (epoch: number) =>
    new Date(epoch).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      timeZone: 'UTC',
    })


  const contentForType = usePlanContent()[type]
  if (!('guide' in contentForType)) return null
  const content = contentForType.guide[planParamsNorm.advanced.strategy]

  const variables = {
    numRuns: `${nonPlanParams.numOfSimulationForMonteCarloSampling}`,
    inflationDate: formatDate(marketData.inflation.closingTime),
    inflation: formatPercentage(1)(marketData.inflation.suggestedAnnual),
    historicalExpectedStockReturn: formatPercentage(1)(
      marketData.expectedReturns.stocks.historical,
    ),
    historicalExpectedBondReturn: formatPercentage(1)(
      marketData.expectedReturns.bonds.historical,
    ),
    historicalReturnDataStartMonth: CalendarMonthFns.toStr(
      marketData.historicalReturnsMonthRange.start,
    ),
    historicalReturnDataStartMonthPlus1: CalendarMonthFns.toStr(
      CalendarMonthFns.addMonths(
        marketData.historicalReturnsMonthRange.start,
        1,
      ),
    ),
    historicalReturnDataStartMonthPlus599: CalendarMonthFns.toStr(
      CalendarMonthFns.addMonths(
        marketData.historicalReturnsMonthRange.start,
        599,
      ),
    ),
    historicalReturnDataStartMonthPlus600: CalendarMonthFns.toStr(
      CalendarMonthFns.addMonths(
        marketData.historicalReturnsMonthRange.start,
        600,
      ),
    ),
    historicalReturnDataEndMonthMinus599: CalendarMonthFns.toStr(
      CalendarMonthFns.addMonths(
        marketData.historicalReturnsMonthRange.start,
        -599,
      ),
    ),
    historicalReturnDataEndMonth: CalendarMonthFns.toStr(
      marketData.historicalReturnsMonthRange.end,
    ),
    tipsYield20Year: formatPercentage(1)(
      marketData.expectedReturns.bonds.tipsYield20Year,
    ),
  }
  return Contentful.replaceVariables(variables, content)
}
