import {
  annualToMonthlyReturnRate,
  MarketData,
  SUGGESTED_ANNUAL_INFLATION,
} from '@tpaw/common'
import { noCase } from '../../Utils/Utils'
import { PlanParamsExtended } from '../ExtentPlanParams'
import { planParamsProcessAdjustmentsToSpending } from './PlanParamsProcessAdjustmentsToSpending'
import { planParamsProcessByMonthParams } from './PlanParamsProcessByMonthParams'
import { planParamsProcessExpectedAnnualReturnForPlanning } from './PlanParamsProcessExpectedAnnualReturnForPlanning'
import { planParamsProcessHistoricalReturnsAdjustment } from './PlanParamsProcessHistoricalReturnsAdjustment'
import { planParamsProcessNetPresentValue } from './PlanParamsProcessNetPresentValue'
import { planParamsProcessRisk } from './PlanParamsProcessRisk'

export type PlanParamsProcessed = ReturnType<typeof processPlanParams>
export function processPlanParams(
  planParamsExt: PlanParamsExtended,
  estimatedCurrentPortfolioBalance: number,
  currentMarketData: MarketData.Data[0],
) {
  const { planParams, currentTimestamp, ianaTimezoneName } = planParamsExt

  const expectedReturnsForPlanning =
    planParamsProcessExpectedAnnualReturnForPlanning(
      planParams.advanced.expectedAnnualReturnForPlanning,
      currentMarketData,
    )
  const historicalReturnsAdjusted =
    planParamsProcessHistoricalReturnsAdjustment(
      planParams,
      expectedReturnsForPlanning,
    )

  const monthlyInflation = annualToMonthlyReturnRate(
    planParams.advanced.annualInflation.type === 'suggested'
      ? SUGGESTED_ANNUAL_INFLATION(currentMarketData)
      : planParams.advanced.annualInflation.type === 'manual'
      ? planParams.advanced.annualInflation.value
      : noCase(planParams.advanced.annualInflation),
  )

  const byMonth = planParamsProcessByMonthParams(
    planParamsExt,
    monthlyInflation,
  )

  const adjustmentsToSpending = planParamsProcessAdjustmentsToSpending(
    planParamsExt,
    monthlyInflation,
  )

  const risk = planParamsProcessRisk(
    planParamsExt,
    expectedReturnsForPlanning,
    historicalReturnsAdjusted.monthly.annualStats.estimatedSampledStats.stocks,
  )

  const netPresentValue = planParamsProcessNetPresentValue(
    planParamsExt,
    risk,
    adjustmentsToSpending.tpawAndSPAW.legacy.target,
    byMonth,
    expectedReturnsForPlanning.monthly,
  )

  const result = {
    strategy: planParams.advanced.strategy,
    currentTime: {
      epoch: currentTimestamp,
      zoneName: ianaTimezoneName,
    },
    people: planParams.people,
    estimatedCurrentPortfolioBalance,
    byMonth,
    netPresentValue,
    adjustmentsToSpending,
    risk,
    expectedReturnsForPlanning,
    historicalReturnsAdjusted,
    monthlyInflation,
    sampling: planParams.advanced.sampling,
    samplingBlockSizeForMonteCarlo:
      planParams.advanced.sampling.blockSizeForMonteCarloSampling,
    original: planParams,
  }

  return result
}
