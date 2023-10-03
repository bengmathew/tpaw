import {
  PlanParams,
  PlanParamsChangeAction,
  assert,
  noCase,
} from '@tpaw/common'
import { formatPercentage } from '../../../Utils/FormatPercentage'
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

    case 'setHistoricalReturnsFixedBonds': {
      // --------- // SetHistoricalReturnsFixedBonds // ---------
      const { value } = action
      return {
        render: () =>
          `Set fixed historical returns bonds to ${formatPercentage(1)(value)}`,
      }
    }

    default:
      return processPlanParamsChangeActionCurrent(action)
  }
}