// Embedding index into type instead of expanding to {type:..., index:number}

import {TPAWParams} from '../../../TPAWSimulator/TPAWParams'

// because it helps when adding to useEffect dep list.
export type ChartPanelType =
  | 'spending-total'
  | 'spending-regular'
  | `spending-essential-${number}`
  | `spending-discretionary-${number}`
  | 'portfolio'
  | 'glide-path'
  | 'withdrawal-rate'

export const isChartPanelSpendingEssentialType = (
  x: ChartPanelType
): x is `spending-essential-${number}` => x.startsWith('spending-essential-')

export const chartPanelSpendingEssentialTypeID = (
  x: `spending-essential-${number}`
) => parseInt(x.substring('spending-essential-'.length))

export const isChartPanelSpendingDiscretionaryType = (
  x: ChartPanelType
): x is `spending-discretionary-${number}` =>
  x.startsWith('spending-discretionary-')

export const chartPanelSpendingDiscretionaryTypeID = (
  x: `spending-discretionary-${number}`
) => parseInt(x.substring('spending-discretionary-'.length))

const _checkType = (x: string): x is ChartPanelType =>
  x === 'spending-total' ||
  x === 'spending-regular' ||
  (x.startsWith('spending-extra-') &&
    _isDigits(x.substring('spending-extra-'.length))) ||
  (x.startsWith('spending-discretionary-') &&
    _isDigits(x.substring('spending-discretionary-'.length))) ||
  x === 'portfolio' ||
  x === 'glide-path' ||
  x === 'withdrawal-rate'

export const isChartPanelType = (
  params: TPAWParams,
  type: string
): type is ChartPanelType => {
  if (!_checkType(type)) return false
  if (isChartPanelSpendingEssentialType(type)) {
    const id = chartPanelSpendingEssentialTypeID(type)
    return params.withdrawals.fundedByBonds.find(x => x.id === id) !== undefined
  }
  if (isChartPanelSpendingDiscretionaryType(type)) {
    const id = chartPanelSpendingDiscretionaryTypeID(type)
    return (
      params.withdrawals.fundedByRiskPortfolio.find(x => x.id === id) !==
      undefined
    )
  }
  return true
}

const _isDigits = (x: string) => /^[0-9]+$/.test(x)
