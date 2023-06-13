import { Params } from '@tpaw/common'
import _ from 'lodash'
import { trimAndNullify } from '../../../../Utils/TrimAndNullify'
import { assert, noCase } from '../../../../Utils/Utils'
import {
  PlanChartType,
  isPlanChartSpendingDiscretionaryType,
  isPlanChartSpendingEssentialType,
  planChartSpendingDiscretionaryTypeID,
  planChartSpendingEssentialTypeID,
} from '../PlanChartType'

export const planChartLabel = (
  params: Params,
  panelType: PlanChartType,
  type: 'full' | 'short',
) => {
  switch (panelType) {
    case 'spending-total':
      return {
        label: ['Monthly Spending During Retirement'],
        subLabel: null,
        description: 'Total retirement spending',
        yAxisDescription: _yAxisDescriptionType.realDollarsExplanation,
      }
    case 'spending-general':
      return {
        label: _.compact([
          type === 'full' ? 'Monthly Spending' : undefined,
          `General`,
        ]),
        subLabel: null,
        description: 'General retirement spending',
        yAxisDescription: _yAxisDescriptionType.realDollarsExplanation,
      }
    case 'portfolio':
      return {
        label: ['Portfolio'],
        subLabel: null,
        description: 'Savings portfolio balance',
        yAxisDescription: _yAxisDescriptionType.realDollarsExplanation,
      }
    case 'asset-allocation-savings-portfolio':
      return {
        label: ['Asset Allocation'],
        subLabel: null,
        description: ' Percentage of portfolio invested in stocks',
        yAxisDescription: _yAxisDescriptionType.assetAllocationExplanation,
      }
    case 'asset-allocation-total-portfolio':
      return {
        label: ['Asset Allocation of Total Portfolio'],
        subLabel: null,
        description: ' Percentage of total portfolio invested in stocks',
        yAxisDescription: _yAxisDescriptionType.assetAllocationExplanation,
      }
    case 'withdrawal':
      return {
        label: ['Monthly Withdrawal Rate'],
        subLabel: null,
        description: 'Percentage of portfolio withdrawn for spending',
        yAxisDescription: null,
      }
    default:
      const { essential, discretionary } =
        params.plan.adjustmentsToSpending.extraSpending
      const splitEssentialAndDiscretionary =
        params.plan.advanced.strategy !== 'SWR'
      if (isPlanChartSpendingEssentialType(panelType)) {
        const id = planChartSpendingEssentialTypeID(panelType)
        const index = essential.findIndex((x) => x.id === id)
        assert(index !== -1)

        const spendingLabel = `${
          trimAndNullify(essential[index].label) ?? '<No label>'
        }`

        const label = _.compact([
          type === 'full' ? 'Monthly Spending' : undefined,
          ...(splitEssentialAndDiscretionary
            ? ['Extra', 'Essential']
            : [spendingLabel]),
        ])
        const subLabel = splitEssentialAndDiscretionary ? spendingLabel : null
        return {
          label,
          subLabel,
          description: 'Extra essential spending',
          yAxisDescription: _yAxisDescriptionType.realDollarsExplanation,
        }
      }
      if (isPlanChartSpendingDiscretionaryType(panelType)) {
        const id = planChartSpendingDiscretionaryTypeID(panelType)
        const index = discretionary.findIndex((x) => x.id === id)
        assert(index !== -1)

        const spendingLabel = `${
          trimAndNullify(discretionary[index].label) ?? '<No label>'
        }`

        const label = _.compact([
          type === 'full' ? 'Monthly Spending' : undefined,
          ...(splitEssentialAndDiscretionary
            ? ['Extra', 'Discretionary']
            : [spendingLabel]),
        ])
        const subLabel = splitEssentialAndDiscretionary ? spendingLabel : null
        return {
          label,
          subLabel,
          description: 'Extra discretionary spending',
          yAxisDescription: _yAxisDescriptionType.realDollarsExplanation,
        }
      }
      noCase(panelType)
  }
}

const _yAxisDescriptionType = {
  realDollarsExplanation: {
    notMobile: [
      { type: 'plain' as const, value: 'These dollars are' },
      { type: 'inflation' as const, value: 'adjusted for inflation' },
    ],
    mobile: [
      { type: 'plain' as const, value: 'Dollars' },
      { type: 'inflation' as const, value: 'adjusted for inflation' },
    ],
  },
  assetAllocationExplanation: {
    notMobile: [
      { type: 'plain' as const, value: 'Percentage of portfolio in stocks' },
    ],
    mobile: [
      { type: 'plain' as const, value: 'Percentage of portfolio in stocks' },
    ],
  },
}
