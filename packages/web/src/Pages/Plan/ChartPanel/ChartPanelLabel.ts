import _ from 'lodash'
import {TPAWParams} from '../../../TPAWSimulator/TPAWParams'
import {trimAndNullify} from '../../../Utils/TrimAndNullify'
import {assert, noCase} from '../../../Utils/Utils'
import {
  chartPanelSpendingDiscretionaryTypeID,
  chartPanelSpendingEssentialTypeID,
  ChartPanelType,
  isChartPanelSpendingDiscretionaryType,
  isChartPanelSpendingEssentialType,
} from './ChartPanelType'

export const chartPanelLabel = (
  params: TPAWParams,
  panelType: ChartPanelType | 'sharpe-ratio',
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
      return {label: ['Asset Allocation'], subLabel: null}
    case 'withdrawal-rate':
      return {label: ['Withdrawal Rate'], subLabel: null}
    case 'sharpe-ratio':
      return {label: ['Reward/Risk Ratio'], subLabel: null}
    default:
      if (isChartPanelSpendingEssentialType(panelType)) {
        const id = chartPanelSpendingEssentialTypeID(panelType)
        const index = params.withdrawals.essential.findIndex(x => x.id === id)
        assert(index !== -1)
        const label = _.compact([
          type === 'full' ? 'Spending' : undefined,
          'Extra',
          'Essential',
        ])
        const subLabel = `${
          trimAndNullify(params.withdrawals.essential[index].label) ??
          '<No label>'
        }`
        return {label, subLabel}
      }
      if (isChartPanelSpendingDiscretionaryType(panelType)) {
        const id = chartPanelSpendingDiscretionaryTypeID(panelType)
        const index = params.withdrawals.discretionary.findIndex(
          x => x.id === id
        )
        assert(index !== -1)
        const label = _.compact([
          type === 'full' ? 'Spending' : undefined,
          `Extra`,
          `Discretionary`,
        ])
        const subLabel = `${
          params.withdrawals.discretionary[index].label ?? '<No label>'
        }`
        return {label, subLabel}
      }
      noCase(panelType)
  }
}
