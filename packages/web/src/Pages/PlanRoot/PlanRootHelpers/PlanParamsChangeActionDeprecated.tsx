import { PlanParams, PlanParamsChangeAction, block, noCase } from '@tpaw/common'
import _ from 'lodash'
import { formatPercentage } from '../../../Utils/FormatPercentage'
import { expectedReturnTypeLabelInfo } from '../Plan/PlanInput/PlanInputExpectedReturnsAndVolatility'
import { processPlanParamsChangeActionCurrent } from './PlanParamsChangeAction'

type _ActionFns = {
  render: (
    prevPlanParams: PlanParams,
    planParams: PlanParams,
  ) => React.ReactNode
}

export const processPlanParamsChangeActionDeprecated = (
  action: PlanParamsChangeAction,
): _ActionFns => {
  switch (action.type) {
    // ---------
    // SwitchHistoricalReturns
    // ---------
    case 'switchHistoricalReturns': {
      const { value } = action
      return {
        // This is only accessible in dev.
        render: () => {
          const stocksAndBondsStr = ({
            stocks,
            bonds,
          }: {
            stocks: number
            bonds: number
          }) =>
            `stocks: ${formatPercentage(1)(stocks)}, bonds: ${formatPercentage(
              1,
            )(bonds)}`
          return `Switch historical returns to ${
            value.type === 'adjusted'
              ? value.adjustment.type === 'toExpected'
                ? 'adjusted to expected'
                : value.adjustment.type === 'to' ||
                    value.adjustment.type === 'by'
                  ? `adjusted ${value.adjustment.type} ${stocksAndBondsStr(
                      value.adjustment,
                    )}`
                  : noCase(value.adjustment)
              : value.type === 'fixed'
                ? `fixed at ${stocksAndBondsStr(value)}`
                : value.type === 'unadjusted'
                  ? 'unadjusted'
                  : noCase(value)
          }`
        },
      }
    }

    // ---------
    // SetHistoricalReturnsAdjustForBlockSampling
    // ---------
    case 'setHistoricalReturnsAdjustForBlockSampling': {
      const { value } = action
      return {
        render: () =>
          `${
            value ? 'Set' : 'Unset'
          } correct historical returns for block sampling.`,
      }
    }

    // ---------
    // SetHistoricalReturnsBonds
    // ---------
    case 'setHistoricalReturnsBonds': {
      const { value } = action
      return {
        render: () => {
          return value === 'adjustExpectedToExpectedUsedForPlanning'
            ? 'Bond volatility enabled'
            : value === 'fixedToExpectedUsedForPlanning'
              ? 'Bond volatility disabled'
              : noCase(value)
        },
      }
    }

    // ---------
    // SetHistoricalReturnsFixedStocks
    // ---------
    case 'setHistoricalReturnsFixedStocks': {
      const { value } = action
      return {
        render: () =>
          `Set fixed historical returns stocks to ${formatPercentage(1)(
            value,
          )}`,
      }
    }

    // ---------
    // SetHistoricalReturnsFixedBonds
    // ---------
    case 'setHistoricalReturnsFixedBonds': {
      const { value } = action
      return {
        render: () =>
          `Set fixed historical returns bonds to ${formatPercentage(1)(value)}`,
      }
    }

    // ---------
    // SetHistoricalReturnsStocksDev
    // ---------
    case 'setHistoricalReturnsStocksDev': {
      const { value } = action
      return {
        render: () => `DEV: Set historical returns for stocks`,
      }
    }

    // ---------
    // SetHistoricalReturnsBondsDev
    // ---------
    case 'setHistoricalReturnsBondsDev': {
      const { value } = action
      return {
        render: () => `DEV: Set historical returns for bonds`,
      }
    }

    // ---------
    // SetExpectedReturns2
    // ---------
    case 'setExpectedReturns': {
      const { value } = action
      return {
        render: () => {
          const labelInfo = expectedReturnTypeLabelInfo({
            type: block(() => {
              switch (value.type) {
                case 'suggested':
                  return 'conservativeEstimate,20YearTIPSYield' as const
                case 'oneOverCAPE':
                  return '1/CAPE,20YearTIPSYield' as const
                case 'regressionPrediction':
                  return 'regressionPrediction,20YearTIPSYield' as const
                default:
                  return value.type
              }
            }),
          })
          const label = labelInfo.isSplit
            ? labelInfo.forUndoRedo
            : _.lowerCase(labelInfo.stocksAndBonds)
          return `Set expected returns to ${label}${
            value.type === 'manual'
              ? ` (stocks: ${formatPercentage(1)(
                  value.stocks,
                )}, bonds: ${formatPercentage(1)(value.bonds)})`
              : ''
          }`
        },
      }
    }
    default:
      return processPlanParamsChangeActionCurrent(action)
  }
}
