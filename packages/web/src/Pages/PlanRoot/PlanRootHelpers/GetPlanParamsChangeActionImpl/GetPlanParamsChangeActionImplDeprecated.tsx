import {
  CalendarMonthFns,
  LabeledAmountTimed,
  LabeledAmountUntimed,
  PlanParams,
  PlanParamsChangeAction,
  block,
  fGet,
  noCase,
} from '@tpaw/common'
import _ from 'lodash'
import { PlanParamsHelperFns } from '../../../../Simulator/PlanParamsHelperFns'
import { formatCurrency } from '../../../../Utils/FormatCurrency'
import { formatPercentage } from '../../../../Utils/FormatPercentage'
import { InMonthsFns } from '../../../../Utils/InMonthsFns'
import { yourOrYourPartners } from '../../../../Utils/YourOrYourPartners'
import { optGet } from '../../../../Utils/optGet'
import { getExpectedReturnTypeLabelInfo } from '../../Plan/PlanInput/PlanInputExpectedReturnsAndVolatilityFns'
import {
  getLabelStrTruncated,
  getLabeledAmountTimedLocationStr,
  getLabeledAmountTimedOrUntimedLocationStr,
  getPlanParamsChangeActionImpl,
} from './GetPlanParamsChangeActionImpl'

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
    // SetExpectedReturns
    // ---------
    case 'setExpectedReturns': {
      const { value } = action
      return {
        render: () => {
          const labelInfo = getExpectedReturnTypeLabelInfo({
            type: block(() => {
              switch (value.type) {
                case 'suggested':
                  return 'conservativeEstimate,20YearTIPSYield'
                case 'oneOverCAPE':
                  return '1/CAPE,20YearTIPSYield'
                case 'regressionPrediction':
                  return 'regressionPrediction,20YearTIPSYield'
                case 'manual':
                  return 'fixed'
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
    // ---------
    // setHistoricalReturnsAdjustExpectedReturnDev
    // ---------
    case 'setHistoricalReturnsAdjustExpectedReturnDev': {
      const { type } = action.value
      return {
        render: () =>
          `DEV: Set historical returns expected value adjustment for ${
            type === 'stocks' ? 'stocks' : 'bonds'
          }`,
      }
    }

    // ---------
    // AddValueForMonthRange
    // ---------
    case 'addValueForMonthRange': {
      const { location } = action.value
      return {
        render: () => {
          return `Added ${getLabeledAmountTimedLocationStr(location)} entry`
        },
      }
    }
    // ---------
    // DeleteLabeledAmount
    // ---------
    case 'deleteLabeledAmount': {
      const { location, entryId } = action.value
      return {
        render: (prevParams) => {
          const prevEntries: Record<
            string,
            LabeledAmountTimed | LabeledAmountUntimed
          > =
            PlanParamsHelperFns.getLabeledAmountTimedOrUntimedListFromLocation(
              prevParams,
              location,
            )
          const prevEntry = optGet(prevEntries, entryId)
          // if (!prevEntry) {
          //   Sentry.captureException(
          //     new Error('No entry for deleteLabeledAmount'),
          //   )
          // }
          const label = prevEntry ? getLabelStrTruncated(prevEntry.label) : ''
          return `Deleted ${getLabeledAmountTimedOrUntimedLocationStr(
            location,
          )} entry ${label}`
        },
      }
    }
    // ---------
    // AddLabeledAmount
    // ---------
    case 'addLabeledAmount': {
      const { location } = action.value
      return {
        render: () => {
          return `Added ${getLabeledAmountTimedOrUntimedLocationStr(location)} entry `
        },
      }
    }

    // ---------
    // SetLabelForLabeledAmount
    // ---------
    case 'setLabelForLabeledAmount': {
      const { location, label } = action.value
      return {
        render: () => {
          return `Set ${getLabeledAmountTimedOrUntimedLocationStr(
            location,
          )} entry label to "${getLabelStrTruncated(label)}'" `
        },
      }
    }

    // ---------
    // setAmountForLabeledAmount
    // ---------
    case 'setAmountForLabeledAmount': {
      const { location, entryId, amount } = action.value
      return {
        render: (_: PlanParams, planParams: PlanParams) => {
          const entries: Record<
            string,
            LabeledAmountUntimed | LabeledAmountTimed
          > =
            PlanParamsHelperFns.getLabeledAmountTimedOrUntimedListFromLocation(
              planParams,
              location,
            )
          return `Set ${getLabeledAmountTimedOrUntimedLocationStr(
            location,
          )} entry "${getLabelStrTruncated(
            fGet(optGet(entries, entryId)).label,
          )}" amount to ${formatCurrency(amount)} per month`
        },
      }
    }

    // ---------
    // SetNominalForLabeledAmount
    // ---------
    case 'setNominalForLabeledAmount': {
      const { location, entryId, nominal } = action.value
      return {
        render: (_: PlanParams, planParams: PlanParams) => {
          const entries: Record<
            string,
            LabeledAmountTimed | LabeledAmountUntimed
          > =
            PlanParamsHelperFns.getLabeledAmountTimedOrUntimedListFromLocation(
              planParams,
              location,
            )

          return `Set ${getLabeledAmountTimedOrUntimedLocationStr(
            location,
          )} entry "${getLabelStrTruncated(
            fGet(optGet(entries, entryId)).label,
          )}" amount as ${nominal ? 'nominal' : 'rea'} dollars`
        },
      }
    }

    // ---------
    // SetMonthRangeForValueForMonthRange
    // ---------
    case 'setMonthRangeForValueForMonthRange': {
      const { location, entryId } = action.value
      return {
        render: (_: PlanParams, planParams: PlanParams) => {
          return `Set ${getLabeledAmountTimedLocationStr(
            location,
          )} entry "${getLabelStrTruncated(
            fGet(
              optGet(
                PlanParamsHelperFns.getLabeledAmountTimedListFromLocation(
                  planParams,
                  location,
                ),
                entryId,
              ),
            ).label,
          )}" month range`
        },
      }
    }

    // ---------
    // SetPersonMonthOfBirth
    // ---------
    case 'setPersonMonthOfBirth': {
      const { person, monthOfBirth } = action.value
      return {
        render: () =>
          `Set ${yourOrYourPartners(
            person,
          )} month of birth to ${CalendarMonthFns.toStr(monthOfBirth)}`,
      }
    }
    // ---------
    // SetMonteCarloSamplingBlockSize
    // ---------
    case 'setMonteCarloSamplingBlockSize': {
      const { value } = action
      return {
        render: () =>
          `Set block size for Monte Carlo simulation to ${InMonthsFns.toStr({ inMonths: value })}`,
      }
    }

    // ---------
    // setHistoricalStockReturnsAdjustmentVolatilityScale
    // ---------
    case 'setHistoricalStockReturnsAdjustmentVolatilityScale': {
      const { value } = action
      return {
        render: () => `Set stocks volatility scaling to ${value}`,
      }
    }

    // ---------
    // setHistoricalBondReturnsAdjustmentEnableVolatility
    // ---------
    case 'setHistoricalBondReturnsAdjustmentEnableVolatility': {
      const { value } = action
      return {
        render: () => `Bond volatility ${value ? 'enabled' : 'disabled'}`,
      }
    }
    // ---------
    // SetExpectedReturns2
    // ---------
    case 'setExpectedReturns2': {
      const { value } = action
      return {
        render: () => {
          const labelInfo = getExpectedReturnTypeLabelInfo({
            type: value.type === 'manual' ? 'fixed' : value.type,
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
    // ---------
    // setPersonMonthOfBirth2
    // ---------
    case 'setPersonMonthOfBirth2': {
      const { person, monthOfBirth } = action.value
      return {
        render: () =>
          `Set ${yourOrYourPartners(
            person,
          )} month of birth to ${CalendarMonthFns.toStr(monthOfBirth)}`,
      }
    }
    // ---------
    // setSPAWAndSWRAllocation
    // ---------
    case 'setSPAWAndSWRAllocation': {
      return {
        render: () => `Updated stock allocation`,
      }
    }
    // ---------
    // setMonthRangeForLabeledAmountTimed
    // ---------
    case 'setMonthRangeForLabeledAmountTimed': {
      const { location, entryId, monthRange } = action.value
      return {
        render: (_: PlanParams, planParams: PlanParams) => {
          return `Set ${getLabeledAmountTimedLocationStr(
            location,
          )} entry "${getLabelStrTruncated(
            fGet(
              optGet(
                PlanParamsHelperFns.getLabeledAmountTimedListFromLocation(
                  planParams,
                  location,
                ),
                entryId,
              ),
            ).label,
          )}" month range`
        },
      }
    }
    // ---------
    // addLabeledAmountTimed
    // ---------
    case 'addLabeledAmountTimed': {
      const { location } = action.value
      return {
        render: () => {
          return `Added ${getLabeledAmountTimedLocationStr(location)} entry`
        },
      }
    }

    // ---------
    // setHistoricalMonthlyLogReturnsAdjustmentOverrideToFixedForTesting
    // ---------
    case 'setHistoricalMonthlyLogReturnsAdjustmentOverrideToFixedForTesting': {
      const { value } = action
      return {
        render: () =>
          `Set historical monthly log returns adjustment override to fixed for testing to ${value}`,
      }
    }

    default:
      return getPlanParamsChangeActionImpl(action)
  }
}
