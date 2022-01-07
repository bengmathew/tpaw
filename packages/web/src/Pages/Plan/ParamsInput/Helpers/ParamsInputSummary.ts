import { TPAWParams } from '../../../../TPAWSimulator/TPAWParams'
import { formatCurrency } from '../../../../Utils/FormatCurrency'
import { formatPercentage } from '../../../../Utils/FormatPercentage'
import { noCase } from '../../../../Utils/Utils'
import { ParamsInputTypes } from './ParamsInputTypes'

export const paramsInputSummary = (
  type: ParamsInputTypes,
  params: TPAWParams
) => {
  switch (type) {
    case 'age': {
      const retired = params.age.start === params.age.retirement
      return retired
        ? `Retired, Current: ${params.age.start}, Max: ${params.age.end}`
        : `Current: ${params.age.start}, Retirement: ${params.age.retirement}, Max: ${params.age.end}`
    }
    case 'currentPortfolioValue': {
      return `${formatCurrency(params.savingsAtStartOfStartYear)}`
    }
    case 'futureSavings': {
      const numEntries = params.savings.length
      return `${numEntries} ${numEntries === 1 ? 'entry' : 'entries'}`
    }
    case 'retirementIncome': {
      const numEntries = params.retirementIncome.length
      return `${numEntries} ${numEntries === 1 ? 'entry' : 'entries'}`
    }
    case 'extraSpending': {
      const numEntries =
        params.withdrawals.fundedByBonds.length +
        params.withdrawals.fundedByRiskPortfolio.length
      return `${numEntries} ${numEntries === 1 ? 'entry' : 'entries'}`
    }
    case 'spendingCeiling': {
      return `${
        params.spendingCeiling === null
          ? 'No ceiling'
          : formatCurrency(params.spendingCeiling)
      }`
    }
    case 'legacy': {
      return `${formatCurrency(params.legacy.total)}`
    }
    case 'expectedReturnsAndInflation': {
      const format = formatPercentage(1)
      return `Stocks: ${format(
        params.returns.expected.stocks
      )}, Bonds: ${format(params.returns.expected.bonds)}, Inflation: ${format(
        params.inflation
      )}`
    }
    default:
      noCase(type)
  }
}
