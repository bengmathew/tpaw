import { noCase } from '../../../../../Utils/Utils'
import { PlanSectionName } from './PlanSectionName'

export function planSectionLabel(type: PlanSectionName): string {
  switch (type) {
    case 'summary':
      return 'Summary'
    case 'age':
      return 'Age'
    case 'current-portfolio-balance':
      return 'Current Portfolio Balance'
    case 'future-savings':
      return 'Future Savings'
    case 'income-during-retirement':
      return 'Income During Retirement'
    case 'help':
      return 'Understanding these Results'
    case 'extra-spending':
      return 'Extra Spending'
    case 'legacy':
      return 'Legacy'
    case 'risk':
      return 'Risk'
    case 'spending-ceiling-and-floor':
      return 'Spending Ceiling and Floor'
    case 'inflation':
      return 'Inflation'
    case 'strategy':
      return 'Strategy'
    case 'expected-returns-and-volatility':
      return 'Expected Returns and Volatility'
    case 'simulation':
      return 'Simulation'
    case 'dev-misc':
      return 'Misc'
    case 'dev-simulations':
      return 'Simulations'
    case 'dev-time':
      return 'Time'
    default:
      noCase(type)
  }
}
