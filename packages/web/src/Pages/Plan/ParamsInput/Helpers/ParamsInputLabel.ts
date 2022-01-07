import {noCase} from '../../../../Utils/Utils'
import {ParamsInputTypes} from './ParamsInputTypes'

export const paramsInputLabel = (type: ParamsInputTypes) => {
  switch (type) {
    case 'age':
      return 'Age'
    case 'currentPortfolioValue':
      return 'Current Portfolio Value'
    case 'futureSavings':
      return 'Future Savings'
    case 'retirementIncome':
      return 'Income During Retirement'
    case 'extraSpending':
      return 'Extra Spending'
    case 'spendingCeiling':
      return 'Spending Ceiling'
    case 'legacy':
      return 'Legacy'
    case 'expectedReturnsAndInflation':
      return 'Expected Returns and Inflation'
    default:
      noCase(type)
  }
}
