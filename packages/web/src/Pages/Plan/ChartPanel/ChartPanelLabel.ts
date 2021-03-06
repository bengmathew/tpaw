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
    case 'asset-allocation-savings-portfolio':
      return {label: ['Asset Allocation of Savings Portfolio'], subLabel: null}
    case 'asset-allocation-total-portfolio':
      return {label: ['Asset Allocation of Total Portfolio'], subLabel: null}
    case 'withdrawal':
      return {label: ['Withdrawal Rate'], subLabel: null}
    case 'sharpe-ratio':
      return {label: ['Reward/Risk Ratio'], subLabel: null}
    default:
      const {essential, discretionary} = params.withdrawals
      const showLabel = params.strategy !== 'SWR'
      if (isChartPanelSpendingEssentialType(panelType)) {
        const id = chartPanelSpendingEssentialTypeID(panelType)
        const index = essential.findIndex(x => x.id === id)
        assert(index !== -1)

        const spendingLabel = `${
          trimAndNullify(essential[index].label) ?? '<No label>'
        }`

        const label = _.compact([
          type === 'full' ? 'Spending' : undefined,
          ...(showLabel
            ? ['Extra', showLabel ? 'Essential' : undefined]
            : [spendingLabel]),
        ])
        const subLabel = showLabel ? spendingLabel : null
        return {label, subLabel}
      }
      if (isChartPanelSpendingDiscretionaryType(panelType)) {
        const id = chartPanelSpendingDiscretionaryTypeID(panelType)
        const index = discretionary.findIndex(x => x.id === id)
        assert(index !== -1)
        
        const spendingLabel = `${
          trimAndNullify(discretionary[index].label) ?? '<No label>'
        }`

        const label = _.compact([
          type === 'full' ? 'Spending' : undefined,
          ...(showLabel
            ? ['Extra', showLabel ? 'Discretionary' : undefined]
            : [spendingLabel]),
        ])
        const subLabel = showLabel ? spendingLabel : null
        return {label, subLabel}
      }
      noCase(panelType)
  }
}
