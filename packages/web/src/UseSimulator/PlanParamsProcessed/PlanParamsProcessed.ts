import { assertFalse } from '@tpaw/common'
import * as Rust from '@tpaw/simulator'
import { PlanParamsNormalized } from '../NormalizePlanParams/NormalizePlanParams'
import { CallRust } from './CallRust'
import { planParamsProcessAdjustmentsToSpending } from './PlanParamsProcessAdjustmentsToSpending'
import { planParamsProcessNetPresentValue } from './PlanParamsProcessNetPresentValue'
import { planParamsProcessRisk } from './PlanParamsProcessRisk'

import jsonpatch from 'fast-json-patch'
import { deepCompare } from '../../Utils/DeepCompare'
import * as Sentry from '@sentry/nextjs'

export type PlanParamsProcessed = ReturnType<typeof processPlanParams>
export function processPlanParams(
  planParamsNorm: PlanParamsNormalized,
  currentMarketData: Rust.DataForMarketBasedPlanParamValues,
) {
  const {
    // marketData,
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

  {
    const riskJS = planParamsProcessRisk(
      planParamsNorm,
      returnsStatsForPlanning,
    )
    if (!deepCompare(riskJS, risk, 0.0000001)) {
      Sentry.captureException(
        new Error(
          `riskJS !== risk\n${JSON.stringify(jsonpatch.compare(risk, riskJS))}\n${JSON.stringify(jsonpatch.compare(riskJS, risk))}`,
        ),
      )
      assertFalse()
    }
  }

  const netPresentValue = planParamsProcessNetPresentValue(
    planParamsNorm.ages.simulationMonths.numMonths,
    risk,
    adjustmentsToSpending.tpawAndSPAW.legacy.target,
    byMonth,
    returnsStatsForPlanning,
  )

  const result = {
    // marketData,
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
