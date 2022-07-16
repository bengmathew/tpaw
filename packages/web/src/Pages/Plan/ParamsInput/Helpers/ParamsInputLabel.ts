import {noCase} from '../../../../Utils/Utils'
import {ParamsInputType} from './ParamsInputType'

export const paramsInputLabel = (type: ParamsInputType) => {
  switch (type) {
    case 'age-and-retirement':
      return 'Age and Retirement'
    case 'current-portfolio-balance':
      return 'Current Portfolio Balance'
    case 'future-savings':
      return 'Future Savings'
    case 'income-during-retirement':
      return 'Income During Retirement'
    case 'extra-spending':
      return 'Extra Spending'
    case 'legacy':
      return 'Legacy'
    case 'stock-allocation':
      return 'Stock Allocation'
    case 'spending-ceiling-and-floor':
      return 'Spending Ceiling and Floor'
    case 'spending-tilt':
      return 'Spending Tilt'
    case 'lmp':
      return 'LMP'
    case 'withdrawal':
      return 'Withdrawal'
    case 'inflation':
      return 'Inflation'
    case 'compare-strategies':
      return 'Compare Strategies'
    case 'expected-returns':
      return 'Expected Returns'
    case 'simulation':
      return 'Simulation'
    case 'dev':
      return 'Dev'
    default:
      noCase(type)
  }
}
