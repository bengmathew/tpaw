import {
  annualToMonthlyReturnRate,
  MarketData,
  SUGGESTED_ANNUAL_INFLATION,
} from '@tpaw/common'
import { noCase } from '../../Utils/Utils'
import { PlanParamsExtended } from '../ExtentPlanParams'
import { planParamsProcessAdjustmentsToSpending } from './PlanParamsProcessAdjustmentsToSpending'
import { planParamsProcessAnnualReturnsParams } from './PlanParamsProcessAnnualReturns'
import { planParamsProcessByMonthParams } from './PlanParamsProcessByMonthParams'
import { planParamsProcessNetPresentValue } from './PlanParamsProcessNetPresentValue'
import { planParamsProcessRisk } from './PlanParamsProcessRisk'

export type PlanParamsProcessed = ReturnType<typeof processPlanParams>
export function processPlanParams(
  planParamsExt: PlanParamsExtended,
  estimatedCurrentPortfolioBalance: number,
  currentMarketData: MarketData.Data[0],
) {
  const { planParams, currentTimestamp, ianaTimezoneName } = planParamsExt

  const returns = planParamsProcessAnnualReturnsParams(
    planParams,
    currentMarketData,
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
    returns.expectedAnnualReturns,
  )

  const netPresentValue = planParamsProcessNetPresentValue(
    planParamsExt,
    risk,
    adjustmentsToSpending.tpawAndSPAW.legacy.target,
    byMonth,
    annualToMonthlyReturnRate(returns.expectedAnnualReturns),
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
    returns,
    monthlyInflation,
    sampling: planParams.advanced.sampling,
    samplingBlockSizeForMonteCarlo:
      planParams.advanced.monteCarloSampling.blockSize,
    original: planParams,
  }

  return result
}
