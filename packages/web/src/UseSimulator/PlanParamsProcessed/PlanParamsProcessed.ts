import * as Rust from '@tpaw/simulator'
import { PlanParamsExtended } from '../ExtentPlanParams'
import { normalizePlanParams } from '../NormalizePlanParams'
import { CallRust } from './CallRust'
import { planParamsProcessAdjustmentsToSpending } from './PlanParamsProcessAdjustmentsToSpending'
import { planParamsProcessNetPresentValue } from './PlanParamsProcessNetPresentValue'
import { planParamsProcessRisk } from './PlanParamsProcessRisk'

export type PlanParamsProcessed = ReturnType<typeof processPlanParams>
export function processPlanParams(
  planParamsExt: PlanParamsExtended,
  estimatedCurrentPortfolioBalance: number,
  currentMarketData: Rust.DataForMarketBasedPlanParamValues,
) {
  const { planParams, currentTimestamp, ianaTimezoneName, numMonths } =
    planParamsExt

  const planParamsNorm = normalizePlanParams(planParamsExt)
  const {
    expectedReturnsForPlanning,
    historicalMonthlyReturnsAdjusted,
    inflation,
    byMonth,
  } = CallRust.processPlanParams(planParamsNorm, currentMarketData)

  const adjustmentsToSpending = planParamsProcessAdjustmentsToSpending(
    planParamsNorm,
    numMonths,
    inflation.monthly,
  )

  const risk = planParamsProcessRisk(
    planParamsExt,
    expectedReturnsForPlanning,
    historicalMonthlyReturnsAdjusted.stocks.stats.empiricalAnnualLogVariance,
  )

  const netPresentValue = planParamsProcessNetPresentValue(
    numMonths,
    risk,
    adjustmentsToSpending.tpawAndSPAW.legacy.target,
    byMonth,
    expectedReturnsForPlanning,
  )

  const result = {
    currentTime: {
      epoch: currentTimestamp,
      zoneName: ianaTimezoneName,
    },
    estimatedCurrentPortfolioBalance,
    byMonth,
    netPresentValue,
    adjustmentsToSpending,
    risk,
    expectedReturnsForPlanning,
    historicalMonthlyReturnsAdjusted,
    inflation,
    planParams,
    planParamsNorm,
  }

  return result
}
