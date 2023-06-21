import {
  annualToMonthlyReturnRate,
  SUGGESTED_ANNUAL_INFLATION,
} from '@tpaw/common'
import { MarketData } from '../../Pages/Common/GetMarketData'
import { noCase } from '../../Utils/Utils'
import { ParamsExtended } from '../ExtentParams'
import { planParamsProcessAdjustmentsToSpending } from './PlanParamsProcessAdjustmentsToSpending'
import { planParamsProcessAnnualReturnsParams } from './PlanParamsProcessAnnualReturns'
import { planParamsProcessByMonthParams } from './PlanParamsProcessByMonthParams'
import { planParamsProcessNetPresentValue } from './PlanParamsProcessNetPresentValue'
import { planParamsProcessRisk } from './PlanParamsProcessRisk'

export type PlanParamsProcessed = ReturnType<typeof processPlanParams>
export function processPlanParams(
  paramsExt: ParamsExtended,
  estimatedCurrentPortfolioBalance: number,
  marketData: MarketData['latest'],
) {
  const { params, currentTime } = paramsExt

  const returns = planParamsProcessAnnualReturnsParams(params, marketData)
  const monthlyInflation = annualToMonthlyReturnRate(
    params.plan.advanced.annualInflation.type === 'suggested'
      ? SUGGESTED_ANNUAL_INFLATION(marketData)
      : params.plan.advanced.annualInflation.type === 'manual'
      ? params.plan.advanced.annualInflation.value
      : noCase(params.plan.advanced.annualInflation),
  )

  const byMonth = planParamsProcessByMonthParams(paramsExt, monthlyInflation)
  const adjustmentsToSpending = planParamsProcessAdjustmentsToSpending(
    paramsExt,
    monthlyInflation,
  )
  const risk = planParamsProcessRisk(paramsExt, returns.expectedAnnualReturns)

  const netPresentValue = planParamsProcessNetPresentValue(
    paramsExt,
    risk,
    adjustmentsToSpending.tpawAndSPAW.legacy.target,
    byMonth,
    annualToMonthlyReturnRate(returns.expectedAnnualReturns),
  )

  const result = {
    strategy: params.plan.advanced.strategy,
    currentTime: {
      epoch: currentTime.valueOf(),
      zoneName: currentTime.zoneName,
    },
    people: params.plan.people,
    estimatedCurrentPortfolioBalance,
    byMonth,
    netPresentValue,
    adjustmentsToSpending,
    risk,
    returns,
    monthlyInflation,
    sampling: params.plan.advanced.sampling,
    samplingBlockSizeForMonteCarlo:
      params.plan.advanced.monteCarloSampling.blockSize,
    original: params,
  }
  return result
}
