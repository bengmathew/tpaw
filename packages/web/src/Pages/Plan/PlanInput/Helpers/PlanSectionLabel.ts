import {PlanParams} from '@tpaw/common'
import {noCase} from '../../../../Utils/Utils'
import {PlanSectionName} from './PlanSectionName'

export function planSectionLabel(
  type: Exclude<PlanSectionName, 'stock-allocation'>
): string
export function planSectionLabel(
  type: PlanSectionName,
  strategy: PlanParams['strategy']
): string
export function planSectionLabel(
  type: PlanSectionName,
  strategy?: PlanParams['strategy']
): string {
  switch (type) {
    case 'summary':
      return 'Summary'
    case 'welcome':
      return 'Welcome'
    case 'age':
      return 'Age'
    case 'current-portfolio-balance':
      return 'Current Portfolio Balance'
    case 'future-savings':
      return 'Future Savings'
    case 'income-during-retirement':
      return 'Income During Retirement'
    case 'results':
      return 'Understanding these Results'
    case 'extra-spending':
      return 'Extra Spending'
    case 'legacy':
      return 'Legacy'
    case 'risk-level':
      return 'Risk Level'
    case 'stock-allocation':
      switch (strategy) {
        case 'TPAW':
          return 'Stock Allocation of Total Portfolio'
        case 'SPAW':
        case 'SWR':
          return 'Stock Allocation of Savings Portfolio'
        case undefined:
          throw new Error()
        default:
          noCase(strategy)
      }
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
    case 'strategy':
      return 'Strategy'
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
