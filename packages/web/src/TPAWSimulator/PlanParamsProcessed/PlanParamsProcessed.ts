import {
  annualToMonthlyReturnRate,
  SUGGESTED_ANNUAL_INFLATION,
} from '@tpaw/common'
import { MarketData } from '../../Pages/Common/GetMarketData'
import { noCase } from '../../Utils/Utils'
import { PlanParamsExt } from '../PlanParamsExt'
import { planParamsProcessAdjustmentsToSpending } from './PlanParamsProcessAdjustmentsToSpending'
import { planParamsProcessAnnualReturnsParams } from './PlanParamsProcessAnnualReturns'
import { planParamsProcessByMonthParams } from './PlanParamsProcessByMonthParams'
import { planParamsProcessRisk } from './PlanParamsProcessRisk'

export type PlanParamsProcessed = ReturnType<typeof processPlanParams>
export function processPlanParams(
  paramsExt: PlanParamsExt,
  marketData: MarketData,
) {
  const { params } = paramsExt

  const returns = planParamsProcessAnnualReturnsParams(
    params,
    marketData,
  )
  const monthlyInflation = annualToMonthlyReturnRate(
    params.advanced.annualInflation.type === 'suggested'
      ? SUGGESTED_ANNUAL_INFLATION(marketData)
      : params.advanced.annualInflation.type === 'manual'
      ? params.advanced.annualInflation.value
      : noCase(params.advanced.annualInflation),
  )

  const result = {
    strategy: params.advanced.strategy,
    people: params.people,
    currentPortfolioBalance: params.wealth.currentPortfolioBalance,
    byMonth: planParamsProcessByMonthParams(paramsExt, monthlyInflation),
    adjustmentsToSpending: planParamsProcessAdjustmentsToSpending(
      paramsExt,
      monthlyInflation,
    ),

    risk: planParamsProcessRisk(paramsExt, returns.expectedAnnualReturns),
    returns,
    monthlyInflation,
    sampling: params.advanced.sampling,
    samplingBlockSizeForMonteCarlo:
      params.advanced.samplingBlockSizeForMonteCarlo,
    original: params,
  }
  return result
}
