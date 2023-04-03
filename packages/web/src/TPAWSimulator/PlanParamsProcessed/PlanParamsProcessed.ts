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

  const result = {
    strategy: params.plan.advanced.strategy,
    currentTime: {
      epoch: currentTime.valueOf(),
      zoneName: currentTime.zoneName,
    },
    people: params.plan.people,
    estimatedCurrentPortfolioBalance,
    byMonth: planParamsProcessByMonthParams(paramsExt, monthlyInflation),
    adjustmentsToSpending: planParamsProcessAdjustmentsToSpending(
      paramsExt,
      monthlyInflation,
    ),

    risk: planParamsProcessRisk(paramsExt, returns.expectedAnnualReturns),
    returns,
    monthlyInflation,
    sampling: params.plan.advanced.sampling,
    samplingBlockSizeForMonteCarlo:
      params.plan.advanced.monteCarloSampling.blockSize,
    original: params,
  }
  return result
}
