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
    case 'spending-ceiling-and-floor':
      return 'Spending Ceiling and Floor'
    case 'legacy':
      return 'Legacy'
    case 'risk-and-time-preference':
      return 'Risk and Time Preference'
    case 'inflation':
      return 'Inflation'
    case 'strategy':
      return 'Strategy'
    case 'expected-returns':
      return 'Expected Returns'
    case 'dev':
      return 'Dev'
    default:
      noCase(type)
  }
}
