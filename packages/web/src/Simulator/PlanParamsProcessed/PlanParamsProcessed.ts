import * as Rust from '@tpaw/simulator'
import { PlanParamsNormalized } from '../NormalizePlanParams/NormalizePlanParams'
import { CallRust } from './CallRust'
import { planParamsProcessAdjustmentsToSpending } from './PlanParamsProcessAdjustmentsToSpending'
import { planParamsProcessNetPresentValue } from './PlanParamsProcessNetPresentValue'


export type PlanParamsProcessed = ReturnType<typeof processPlanParams>
export function processPlanParams(
  planParamsNorm: PlanParamsNormalized,
  currentMarketData: Rust.DataForMarketBasedPlanParamValues,
) {
  const {
    marketData:marketDataProcessed,
    returnsStatsForPlanning,
    historicalReturnsAdjusted,
    inflation,
    risk,
    byMonth,
  } = CallRust.processPlanParams(planParamsNorm, currentMarketData)

  const adjustmentsToSpending = planParamsProcessAdjustmentsToSpending(
    planParamsNorm,
    inflation.monthly,
  )

  const netPresentValue = planParamsProcessNetPresentValue(
    planParamsNorm.ages.simulationMonths.numMonths,
    risk,
    adjustmentsToSpending.tpawAndSPAW.legacy.target,
    byMonth,
    returnsStatsForPlanning,
  )

  const result = {
    marketDataProcessed,
    byMonth,
    netPresentValue,
    adjustmentsToSpending,
    risk,
    returnsStatsForPlanning,
    historicalReturnsAdjusted,
    inflation,
  }

  return result
}
