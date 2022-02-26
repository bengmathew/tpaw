import _ from 'lodash'
import { TPAWParams } from '../../../TPAWSimulator/TPAWParams'
import { assert, noCase } from '../../../Utils/Utils'
import {
  chartPanelSpendingDiscretionaryTypeID,
  chartPanelSpendingEssentialTypeID,
  ChartPanelType,
  isChartPanelSpendingDiscretionaryType,
  isChartPanelSpendingEssentialType
} from './ChartPanelType'

export const chartPanelLabel = (
  params: TPAWParams,
  panelType: ChartPanelType,
  type: 'full' | 'short'
) => {
  switch (panelType) {
    case 'spending-total':
      return {label: ['Spending During Retirement'], subLabel: null}
    case 'spending-general':
      return {
        label: _.compact([type === 'full' ? 'Spending' : undefined, `General`]),
        subLabel: null,
      }
    case 'portfolio':
      return {label: ['Portfolio'], subLabel: null}
    case 'glide-path':
      return {label: ['Glide Path'], subLabel: null}
    case 'withdrawal-rate':
      return {label: ['Withdrawal Rate'], subLabel: null}
    default:
      if (isChartPanelSpendingEssentialType(panelType)) {
        const id = chartPanelSpendingEssentialTypeID(panelType)
        const index = params.withdrawals.fundedByBonds.findIndex(
          x => x.id === id
        )
        assert(index !== -1)
        const label = _.compact([
          type === 'full' ? 'Spending' : undefined,
          'Extra',
          'Essential',
        ])
        const subLabel = `${
          params.withdrawals.fundedByBonds[index].label ?? '<No label>'
        }`
        return {label, subLabel}
      }
      if (isChartPanelSpendingDiscretionaryType(panelType)) {
        const id = chartPanelSpendingDiscretionaryTypeID(panelType)
        const index = params.withdrawals.fundedByRiskPortfolio.findIndex(
          x => x.id === id
        )
        assert(index !== -1)
        const label = _.compact([
          type === 'full' ? 'Spending' : undefined,
          `Extra`,
          `Discretionary`,
        ])
        const subLabel = `${
          params.withdrawals.fundedByRiskPortfolio[index].label ?? '<No label>'
        }`
        return {label, subLabel}
      }
      noCase(panelType)
  }
}
