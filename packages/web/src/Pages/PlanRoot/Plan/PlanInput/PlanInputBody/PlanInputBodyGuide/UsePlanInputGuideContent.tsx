import { CalendarMonthFns } from '@tpaw/common'
import { Contentful } from '../../../../../../Utils/Contentful'
import { useNonPlanParams } from '../../../../PlanRootHelpers/WithNonPlanParams'
import { usePlanContent } from '../../../../PlanRootHelpers/WithPlanContent'
import {
  useSimulationInfo,
  useSimulationResultInfo,
} from '../../../../PlanRootHelpers/WithSimulation'
import { PlanInputType } from '../../Helpers/PlanInputType'
import { formatPercentage } from '../../../../../../Utils/FormatPercentage'

export function usePlanInputGuideContent(type: PlanInputType) {
  const { nonPlanParams } = useNonPlanParams()
  const { planParamsNormInstant } = useSimulationInfo()
  const { planParamsProcessed, numOfSimulationForMonteCarloSamplingOfResult } = useSimulationResultInfo().simulationResult

  const formatDate = (epoch: number) =>
    new Date(epoch).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      timeZone: 'UTC',
    })

  const contentForType = usePlanContent()[type]
  if (!('guide' in contentForType)) return null
  const content = contentForType.guide[planParamsNormInstant.advanced.strategy]

  const variables = {
    numOfSimulationForMonteCarloSampling:
      numOfSimulationForMonteCarloSamplingOfResult.toString(),
    numRuns: `${nonPlanParams.numOfSimulationForMonteCarloSampling}`,
    inflationDate: formatDate(
      planParamsProcessed?.marketDataForPresets.sourceRounded.dailyMarketData
        .inflation.closingTimestamp ?? null,
    ),
    inflation: formatPercentage(1)(
      planParamsProcessed.marketDataForPresets.sourceRounded.dailyMarketData
        .inflation.value,
    ),
    historicalExpectedStockReturn: formatPercentage(1)(
      planParamsProcessed.marketDataForPresets.expectedReturns.stocks
        .historical,
    ),
    historicalExpectedBondReturn: formatPercentage(1)(
      planParamsProcessed.marketDataForPresets.expectedReturns.bonds.historical,
    ),
    historicalReturnDataStartMonth: CalendarMonthFns.toStr(
      planParamsProcessed.historicalReturns.monthRange.start,
    ),
    historicalReturnDataStartMonthPlus1: CalendarMonthFns.toStr(
      CalendarMonthFns.addMonths(
        planParamsProcessed.historicalReturns.monthRange.start,
        1,
      ),
    ),
    historicalReturnDataStartMonthPlus599: CalendarMonthFns.toStr(
      CalendarMonthFns.addMonths(
        planParamsProcessed.historicalReturns.monthRange.start,
        599,
      ),
    ),
    historicalReturnDataStartMonthPlus600: CalendarMonthFns.toStr(
      CalendarMonthFns.addMonths(
        planParamsProcessed.historicalReturns.monthRange.start,
        600,
      ),
    ),
    historicalReturnDataEndMonthMinus599: CalendarMonthFns.toStr(
      CalendarMonthFns.addMonths(
        planParamsProcessed.historicalReturns.monthRange.end,
        -599,
      ),
    ),
    historicalReturnDataEndMonth: CalendarMonthFns.toStr(
      planParamsProcessed.historicalReturns.monthRange.end,
    ),
    tipsYield20Year: formatPercentage(1)(
      planParamsProcessed.marketDataForPresets.expectedReturns.bonds
        .tipsYield20Year,
    ),
  }
  return Contentful.replaceVariables(variables, content)
}
