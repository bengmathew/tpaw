// Embedding index into type instead of expanding to {type:..., index:number}

import { PlanParams } from '@tpaw/common'
import _ from 'lodash'
import { optGet } from '../../../../Utils/optGet'

// because it helps when adding to useEffect dep list.
export type PlanResultsChartType =
  | 'spending-total'
  | 'spending-general'
  | `spending-essential-${string}`
  | `spending-discretionary-${string}`
  | 'portfolio'
  | 'asset-allocation-savings-portfolio'
  | 'asset-allocation-total-portfolio'
  | 'withdrawal'

export const isPlanResultsChartSpendingEssentialType = (
  x: PlanResultsChartType,
): x is `spending-essential-${string}` => x.startsWith('spending-essential-')

export const planResultsChartSpendingEssentialTypeID = (
  x: `spending-essential-${string}`,
) => x.substring('spending-essential-'.length)

export const isPlanResultsChartSpendingDiscretionaryType = (
  x: PlanResultsChartType,
): x is `spending-discretionary-${string}` =>
  x.startsWith('spending-discretionary-')

export const planResultsChartSpendingDiscretionaryTypeID = (
  x: `spending-discretionary-${string}`,
) => x.substring('spending-discretionary-'.length)

const _checkType = (x: string): x is PlanResultsChartType =>
  x === 'spending-total' ||
  x === 'spending-general' ||
  x.startsWith('spending-essential-') ||
  x.startsWith('spending-discretionary-') ||
  x === 'portfolio' ||
  x === 'asset-allocation-savings-portfolio' ||
  x === 'asset-allocation-total-portfolio' ||
  x === 'withdrawal'

export const isPlanResultsChartType = (
  planParams: PlanParams,
  type: string,
): type is PlanResultsChartType => {
  if (!_checkType(type)) return false
  if (isPlanResultsChartSpendingEssentialType(type)) {
    const id = planResultsChartSpendingEssentialTypeID(type)
    return (
      optGet(planParams.adjustmentsToSpending.extraSpending.essential, id) !==
      undefined
    )
  }
  if (isPlanResultsChartSpendingDiscretionaryType(type)) {
    const id = planResultsChartSpendingDiscretionaryTypeID(type)
    const x = _.get(
      planParams.adjustmentsToSpending.extraSpending.discretionary,
      id,
    )
    return (
      optGet(
        planParams.adjustmentsToSpending.extraSpending.discretionary,
        id,
      ) !== undefined
    )
  }
  return true
}

const _isDigits = (x: string) => /^[0-9]+$/.test(x)
