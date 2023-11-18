import { PlanParams, assert, block, fGet, noCase } from '@tpaw/common'
import _ from 'lodash'
import { pluralize } from '../../../../../Utils/Pluralize'
import { trimAndNullify } from '../../../../../Utils/TrimAndNullify'
import { optGet } from '../../../../../Utils/optGet'
import {
  PlanResultsChartType,
  PlanResultsSpendingChartType,
  isPlanResultsChartSpendingDiscretionaryType,
  isPlanResultsChartSpendingEssentialType,
  isPlanResultsChartSpendingTotalFundingSourcesType,
  isPlanResultsChartSpendingType,
  planResultsChartSpendingDiscretionaryTypeID,
  planResultsChartSpendingEssentialTypeID,
} from '../PlanResultsChartType'

export const planResultsChartLabel = (
  planParams: PlanParams,
  panelType: PlanResultsChartType,
) => {
  switch (panelType) {
    case 'portfolio':
      const full = ['Portfolio Balance']
      return {
        label: { full, forMenu: full },
        subLabel: null,
        description: 'Savings portfolio balance',
        yAxisDescription: _yAxisDescriptionType.realDollarsExplanation,
      }
    case 'asset-allocation-savings-portfolio': {
      const full = ['Asset Allocation']
      return {
        label: { full, forMenu: full },
        subLabel: null,
        description: ' Percentage of portfolio invested in stocks',
        yAxisDescription: _yAxisDescriptionType.assetAllocationExplanation,
      }
    }
    case 'asset-allocation-total-portfolio': {
      const full = ['Dev', 'Asset Allocation of Total Portfolio']
      return {
        label: { full, forMenu: full },
        subLabel: null,
        description: ' Percentage of total portfolio invested in stocks',
        yAxisDescription: _yAxisDescriptionType.assetAllocationExplanation,
      }
    }
    case 'withdrawal': {
      const full = ['Monthly Withdrawal Rate']
      return {
        label: { full, forMenu: full },
        subLabel: null,
        description: 'Percentage of portfolio withdrawn for spending',
        yAxisDescription: null,
      }
    }
    default:
      if (isPlanResultsChartSpendingType(panelType)) {
        return getPlanResultsChartLabelInfoForSpending(planParams).getLabelInfo(
          panelType,
        )
      }
      noCase(panelType)
  }
}

export const getPlanResultsChartLabelInfoForSpending = (
  planParams: PlanParams,
) => {
  const { essential, discretionary } =
    planParams.adjustmentsToSpending.extraSpending
  const numEssential = _.values(essential).length
  const numDiscretionary = _.values(discretionary).length
  const numExtra = numEssential + numDiscretionary
  const spendingTotalInfo = block(() => {
    const full = ['Monthly Spending During Retirement']
    return {
      label: { full, forMenu: full },
      subLabel: null,
      description: 'Total retirement spending',
      yAxisDescription: _yAxisDescriptionType.realDollarsExplanation,
    }
  })
  if (numExtra === 0) {
    return {
      hasExtra: false,
      getLabelInfo: (chartType: PlanResultsSpendingChartType) => {
        if (
          isPlanResultsChartSpendingTotalFundingSourcesType(chartType) ||
          chartType === 'spending-total'
        ) {
          return spendingTotalInfo
        }
        assert(
          !(
            chartType === 'spending-general' ||
            isPlanResultsChartSpendingDiscretionaryType(chartType) ||
            isPlanResultsChartSpendingEssentialType(chartType)
          ),
        )
        noCase(chartType)
      },
    } as const
  } else {
    const splitEssentialAndDiscretionary =
      planParams.advanced.strategy !== 'SWR'
    return {
      hasExtra: true,
      extraSpendingLabelInfo: splitEssentialAndDiscretionary
        ? ({
            splitEssentialAndDiscretionary: true,
            essential: {
              label: ['Extra Spending', 'Essential'],
              description: `Extra essential spending (${pluralize(
                numEssential,
                'graph',
              )})`,
            },
            discretionary: {
              label: ['Extra Spending', 'Discretionary'],
              description: `Extra discretionary spending (${pluralize(
                numDiscretionary,
                'graph',
              )})`,
            },
          } as const)
        : ({
            splitEssentialAndDiscretionary: false,
            label: ['Extra Spending'],
            description: `Extra  spending (${pluralize(
              numDiscretionary + numEssential,
              'graph',
            )})`,
          } as const),
      getLabelInfo: (chartType: PlanResultsSpendingChartType) => {
        if (
          isPlanResultsChartSpendingTotalFundingSourcesType(chartType) ||
          chartType === 'spending-total'
        ) {
          return spendingTotalInfo
        }
        if (chartType === 'spending-general') {
          return {
            label: {
              full: ['Monthly Spending', 'General'],
              forMenu: ['General Spending'],
            },
            subLabel: null,
            description: 'Excludes extra spending',
            yAxisDescription: _yAxisDescriptionType.realDollarsExplanation,
          }
        }

        if (isPlanResultsChartSpendingEssentialType(chartType)) {
          const id = planResultsChartSpendingEssentialTypeID(chartType)
          const spendingLabel = `${
            trimAndNullify(fGet(optGet(essential, id)).label) ?? '<No label>'
          }`
          return {
            ...(splitEssentialAndDiscretionary
              ? {
                  label: {
                    full: ['Monthly Spending', 'Extra', 'Essential'],
                    forMenu: [spendingLabel],
                  },
                }
              : {
                  label: {
                    full: ['Monthly Spending', 'Extra'],
                    forMenu: [spendingLabel],
                  },
                }),
            subLabel: spendingLabel,
            description: splitEssentialAndDiscretionary
              ? 'Essential expense'
              : 'Extra expense',
            yAxisDescription: _yAxisDescriptionType.realDollarsExplanation,
          }
        }
        if (isPlanResultsChartSpendingDiscretionaryType(chartType)) {
          const id = planResultsChartSpendingDiscretionaryTypeID(chartType)
          const spendingLabel = `${
            trimAndNullify(fGet(optGet(discretionary, id)).label) ??
            '<No label>'
          }`
          return {
            ...(splitEssentialAndDiscretionary
              ? {
                  label: {
                    full: ['Monthly Spending', 'Extra', 'Discretionary'],
                    forMenu: [spendingLabel],
                  },
                }
              : {
                  label: {
                    full: ['Monthly Spending', 'Extra'],
                    forMenu: [spendingLabel],
                  },
                }),
            subLabel: spendingLabel,
            description: splitEssentialAndDiscretionary
              ? 'Discretionary expense'
              : 'Extra expense',
            yAxisDescription: _yAxisDescriptionType.realDollarsExplanation,
          }
        }
        noCase(chartType)
      },
    }
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
