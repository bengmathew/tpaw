import {
  EXPECTED_ANNUAL_RETURN_PRESETS,
  MarketData,
  PlanParams,
  annualToMonthlyReturnRate,
} from '@tpaw/common'

export const planParamsProcessExpectedAnnualReturnForPlanning = (
  expectedAnnualReturnForPlanning: PlanParams['advanced']['expectedAnnualReturnForPlanning'],
  currentMarketData: MarketData.Data[0],
) => {
  const annual =
    expectedAnnualReturnForPlanning.type === 'manual'
      ? {
          stocks: expectedAnnualReturnForPlanning.stocks,
          bonds: expectedAnnualReturnForPlanning.bonds,
        }
      : EXPECTED_ANNUAL_RETURN_PRESETS(
          expectedAnnualReturnForPlanning.type,
          currentMarketData,
        )

  const monthly = annualToMonthlyReturnRate(annual)
  return { annual, monthly }
}
