import {noCase} from '../../../../Utils/Utils'
import {ParamsInputType} from './ParamsInputType'

export const paramsInputLabel = (type: ParamsInputType) => {
  switch (type) {
    case 'age':
      return 'Age and Retirement'
    case 'risk-and-time-preference':
      return 'Risk and Time Preference'
    case 'current-portfolio-balance':
      return 'Current Portfolio Balance'
    case 'future-savings':
      return 'Future Savings'
    case 'income-during-retirement':
      return 'Income During Retirement'
    case 'extra-spending':
      return 'Extra Spending'
    case 'spending-ceiling-and-floor':
      return 'Spending Ceiling and Floor'
    case 'legacy':
      return 'Legacy'
    case 'expected-returns':
      return 'Expected Returns'
    case 'inflation':
      return 'Inflation'
    default:
      noCase(type)
  }
}
