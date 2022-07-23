// Embedding index into type instead of expanding to {type:..., index:number}

import {TPAWParams} from '../../../TPAWSimulator/TPAWParams'

// because it helps when adding to useEffect dep list.
export type ChartPanelType =
  | 'spending-total'
  | 'spending-general'
  | `spending-essential-${number}`
  | `spending-discretionary-${number}`
  | 'portfolio'
  | 'asset-allocation-savings-portfolio'
  | 'asset-allocation-total-portfolio'
  | 'withdrawal'

export const isChartPanelSpendingEssentialType = (
  x: ChartPanelType | 'sharpe-ratio'
): x is `spending-essential-${number}` => x.startsWith('spending-essential-')

export const chartPanelSpendingEssentialTypeID = (
  x: `spending-essential-${number}`
) => parseInt(x.substring('spending-essential-'.length))

export const isChartPanelSpendingDiscretionaryType = (
  x: ChartPanelType | 'sharpe-ratio'
): x is `spending-discretionary-${number}` =>
  x.startsWith('spending-discretionary-')

export const chartPanelSpendingDiscretionaryTypeID = (
  x: `spending-discretionary-${number}`
) => parseInt(x.substring('spending-discretionary-'.length))

const _checkType = (x: string): x is ChartPanelType =>
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

export const isChartPanelType = (
  params: TPAWParams,
  type: string
): type is ChartPanelType => {
  if (!_checkType(type)) return false
  if (isChartPanelSpendingEssentialType(type)) {
    const id = chartPanelSpendingEssentialTypeID(type)
    return params.withdrawals.essential.find(x => x.id === id) !== undefined
  }
  if (isChartPanelSpendingDiscretionaryType(type)) {
    const id = chartPanelSpendingDiscretionaryTypeID(type)
    return params.withdrawals.discretionary.find(x => x.id === id) !== undefined
  }
  return true
}

const _isDigits = (x: string) => /^[0-9]+$/.test(x)
