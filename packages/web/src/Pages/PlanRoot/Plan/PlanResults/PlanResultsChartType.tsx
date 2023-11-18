import { PlanParams, noCase } from '@tpaw/common'
import _ from 'lodash'
import {
  PERCENTILES_STR,
  Percentile,
} from '../../../../TPAWSimulator/Worker/TPAWRunInWorker'
import { optGet } from '../../../../Utils/optGet'

type SpendingTotalFundingSources =
  `spending-total-funding-sources-${Percentile}`

// Embedding id into type instead of expanding to {type:..., index:number}
// because it helps when adding to useEffect dep list.
export type PlanResultsChartType =
  | PlanResultsSpendingChartType
  | 'portfolio'
  | 'asset-allocation-savings-portfolio'
  | 'asset-allocation-total-portfolio'
  | 'withdrawal'

export type PlanResultsSpendingChartType =
  | 'spending-total'
  | 'spending-general'
  | `spending-essential-${string}`
  | `spending-discretionary-${string}`
  | SpendingTotalFundingSources

export const isPlanResultsChartSpendingType = (
  x: PlanResultsChartType,
): x is PlanResultsSpendingChartType => {
  switch (x) {
    case 'asset-allocation-savings-portfolio':
    case 'asset-allocation-total-portfolio':
    case 'portfolio':
    case 'withdrawal':
      return false
    case 'spending-total':
    case 'spending-general':
      return true
    default:
      if (
        isPlanResultsChartSpendingDiscretionaryType(x) ||
        isPlanResultsChartSpendingEssentialType(x) ||
        isPlanResultsChartSpendingTotalFundingSourcesType(x)
      )
        return true
      noCase(x)
  }
}

export const isPlanResultsChartSpendingTotalFundingSourcesType = (
  x: PlanResultsChartType,
): x is SpendingTotalFundingSources =>
  x.startsWith('spending-total-funding-sources-') &&
  (PERCENTILES_STR as readonly string[]).includes(
    x.substring('spending-total-funding-sources-'.length),
  )
export const getPlanResultsChartSpendingTotalFundingSourcesPercentile = (
  x: SpendingTotalFundingSources,
) => x.substring('spending-total-funding-sources-'.length) as Percentile

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
  x.startsWith('spending-total-funding-sources-') ||
  x === 'portfolio' ||
  x === 'asset-allocation-savings-portfolio' ||
  x === 'asset-allocation-total-portfolio' ||
  x === 'withdrawal'

export const isPlanResultsChartType = (
  planParams: PlanParams,
  type: string,
): type is PlanResultsChartType => {
  if (!_checkType(type)) return false
  if (isPlanResultsChartSpendingType(type)) {
    if (
      type === 'spending-total' ||
      isPlanResultsChartSpendingTotalFundingSourcesType(type)
    )
      return true
    if (type === 'spending-general') {
      const { essential, discretionary } =
        planParams.adjustmentsToSpending.extraSpending
      return _.values(essential).length + _.values(discretionary).length > 0
    }
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
    noCase(type)
  }
  switch (type) {
    case 'portfolio':
    case 'asset-allocation-savings-portfolio':
    case 'asset-allocation-total-portfolio':
    case 'withdrawal':
      return true
    default:
      noCase(type)
  }
}
