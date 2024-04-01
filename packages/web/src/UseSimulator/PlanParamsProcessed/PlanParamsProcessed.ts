import * as Rust from '@tpaw/simulator'
import { PlanParamsNormalized } from '../NormalizePlanParams/NormalizePlanParams'
import { CallRust } from './CallRust'
import { planParamsProcessAdjustmentsToSpending } from './PlanParamsProcessAdjustmentsToSpending'
import { planParamsProcessNetPresentValue } from './PlanParamsProcessNetPresentValue'
import { planParamsProcessRisk } from './PlanParamsProcessRisk'

export type PlanParamsProcessed = ReturnType<typeof processPlanParams>
export function processPlanParams(
  planParamsNorm: PlanParamsNormalized,
  currentMarketData: Rust.DataForMarketBasedPlanParamValues,
) {
  const {
    expectedReturnsForPlanning,
    historicalMonthlyReturnsAdjusted,
    inflation,
    byMonth,
  } = CallRust.processPlanParams(planParamsNorm, currentMarketData)

  const adjustmentsToSpending = planParamsProcessAdjustmentsToSpending(
    planParamsNorm,
    planParamsNorm.ages.simulationMonths.numMonths,
    inflation.monthly,
  )

  const risk = planParamsProcessRisk(
    planParamsNorm,
    expectedReturnsForPlanning,
    historicalMonthlyReturnsAdjusted.stocks.stats.empiricalAnnualLogVariance,
  )

  const netPresentValue = planParamsProcessNetPresentValue(
    planParamsNorm.ages.simulationMonths.numMonths,
    risk,
    adjustmentsToSpending.tpawAndSPAW.legacy.target,
    byMonth,
    expectedReturnsForPlanning,
  )

  const result = {
    byMonth,
    netPresentValue,
    adjustmentsToSpending,
    risk,
    expectedReturnsForPlanning,
    historicalMonthlyReturnsAdjusted,
    inflation,
  }

  return result
}
