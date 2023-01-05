// Embedding index into type instead of expanding to {type:..., index:number}

import { PlanParams } from '@tpaw/common'

// because it helps when adding to useEffect dep list.
export type PlanChartType =
  | 'spending-total'
  | 'spending-general'
  | `spending-essential-${number}`
  | `spending-discretionary-${number}`
  | 'portfolio'
  | 'asset-allocation-savings-portfolio'
  | 'asset-allocation-total-portfolio'
  | 'withdrawal'

export const isPlanChartSpendingEssentialType = (
  x: PlanChartType,
): x is `spending-essential-${number}` => x.startsWith('spending-essential-')

export const planChartSpendingEssentialTypeID = (
  x: `spending-essential-${number}`,
) => parseInt(x.substring('spending-essential-'.length))

export const isPlanChartSpendingDiscretionaryType = (
  x: PlanChartType,
): x is `spending-discretionary-${number}` =>
  x.startsWith('spending-discretionary-')

export const planChartSpendingDiscretionaryTypeID = (
  x: `spending-discretionary-${number}`,
) => parseInt(x.substring('spending-discretionary-'.length))

const _checkType = (x: string): x is PlanChartType =>
  x === 'spending-total' ||
  x === 'spending-general' ||
  (x.startsWith('spending-essential-') &&
    _isDigits(x.substring('spending-essential-'.length))) ||
  (x.startsWith('spending-discretionary-') &&
    _isDigits(x.substring('spending-discretionary-'.length))) ||
  x === 'portfolio' ||
  x === 'asset-allocation-savings-portfolio' ||
  x === 'asset-allocation-total-portfolio' ||
  x === 'withdrawal'

export const isPlanChartType = (
  params: PlanParams,
  type: string,
): type is PlanChartType => {
  if (!_checkType(type)) return false
  if (isPlanChartSpendingEssentialType(type)) {
    const id = planChartSpendingEssentialTypeID(type)
    return (
      params.adjustmentsToSpending.extraSpending.essential.find(
        (x) => x.id === id,
      ) !== undefined
    )
  }
  if (isPlanChartSpendingDiscretionaryType(type)) {
    const id = planChartSpendingDiscretionaryTypeID(type)
    return (
      params.adjustmentsToSpending.extraSpending.discretionary.find(
        (x) => x.id === id,
      ) !== undefined
    )
  }
  return true
}

const _isDigits = (x: string) => /^[0-9]+$/.test(x)
