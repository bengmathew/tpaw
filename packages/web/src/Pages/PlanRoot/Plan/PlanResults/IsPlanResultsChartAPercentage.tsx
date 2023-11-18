import { noCase } from '../../../../Utils/Utils'
import {
  PlanResultsChartType,
  isPlanResultsChartSpendingDiscretionaryType,
  isPlanResultsChartSpendingEssentialType,
  isPlanResultsChartSpendingTotalFundingSourcesType,
} from './PlanResultsChartType'

export const isPlanResultsChartAPercentage = (
  chartType: PlanResultsChartType,
) => {
  switch (chartType) {
    case 'asset-allocation-savings-portfolio':
    case 'asset-allocation-total-portfolio':
    case 'withdrawal':
      return true
    case 'portfolio':
    case 'spending-general':
    case 'spending-total':
      return false
    default:
      if (
        isPlanResultsChartSpendingTotalFundingSourcesType(chartType) ||
        isPlanResultsChartSpendingEssentialType(chartType) ||
        isPlanResultsChartSpendingDiscretionaryType(chartType)
      )
        return false
      noCase(chartType)
  }
}
