import { CalendarMonthFns } from '@tpaw/common'
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
  const { planParamsNorm, currentMarketData } = useSimulation()
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
      CallRust.getPlanParamsRust(planParamsNorm).advanced.sampling,
      planParamsNorm.advanced.historicalMonthlyLogReturnsAdjustment
        .standardDeviation,
      currentMarketData,
    )

  const contentForType = usePlanContent()[type]
  if (!('guide' in contentForType)) return null
  const content = contentForType.guide[planParamsNorm.advanced.strategy]

  const variables = {
    numRuns: `${nonPlanParams.numOfSimulationForMonteCarloSampling}`,
    inflationDate: formatDate(inflation.closingTime),
    inflation: formatPercentage(1)(inflation.value),
    historicalExpectedStockReturn: formatPercentage(1)(
      presetInfo.stocks.historical,
    ),
    historicalExpectedBondReturn: formatPercentage(1)(
      presetInfo.bonds.historical,
    ),
    historicalReturnDataStartMonth: CalendarMonthFns.toStr(
      presetInfo.historicalReturnsMonthRange.start,
    ),
    historicalReturnDataStartMonthPlus1: CalendarMonthFns.toStr(
      CalendarMonthFns.addMonths(
        presetInfo.historicalReturnsMonthRange.start,
        1,
      ),
    ),
    historicalReturnDataStartMonthPlus599: CalendarMonthFns.toStr(
      CalendarMonthFns.addMonths(
        presetInfo.historicalReturnsMonthRange.start,
        599,
      ),
    ),
    historicalReturnDataStartMonthPlus600: CalendarMonthFns.toStr(
      CalendarMonthFns.addMonths(
        presetInfo.historicalReturnsMonthRange.start,
        600,
      ),
    ),
    historicalReturnDataEndMonthMinus599: CalendarMonthFns.toStr(
      CalendarMonthFns.addMonths(
        presetInfo.historicalReturnsMonthRange.start,
        -599,
      ),
    ),
    historicalReturnDataEndMonth: CalendarMonthFns.toStr(
      presetInfo.historicalReturnsMonthRange.end,
    ),
    tipsYield20Year: formatPercentage(1)(
      currentMarketData.bondRates.twentyYear,
    ),
  }
  return Contentful.replaceVariables(variables, content)
}
