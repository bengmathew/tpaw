import {
  assert,
  assertFalse,
  block,
  CalendarDayFns,
  CalendarMonthFns,
  fGet,
  InMonthsFns,
  LabeledAmountTimed,
  LabeledAmountTimedLocation,
  LabeledAmountTimedOrUntimedLocation,
  LabeledAmountUntimed,
  LabeledAmountUntimedLocation,
  noCase,
  optGet,
  PlanParams,
  PlanParamsChangeAction,
  yourOrYourPartners,
} from '@tpaw/common'
import _ from 'lodash'
import { PlanParamsHelperFns } from '@tpaw/common'
import { formatCurrency } from '../../../../../Utils/FormatCurrency'
import { formatPercentage } from '../../../../../Utils/FormatPercentage'
import { planSectionLabel } from '../../PlanInput/Helpers/PlanSectionLabel'
import {
  getExpectedReturnCustomBondBaseLabel,
  getExpectedReturnCustomStockBaseLabel,
  getExpectedReturnTypeLabelInfo,
} from '../../PlanInput/PlanInputExpectedReturnsAndVolatilityFns'
import { inflationTypeLabel } from '../../PlanInput/PlanInputInflationFns'

import { PlanParamsChangeActionDeprecated } from '@tpaw/common/dist/Params/PlanParams/PlanParamsChangeAction/PlanParamsChangeActionDeprecated'

export const renderPlanParamsChangeAction = (
  prevPlanParams: PlanParams,
  planParams: PlanParams,
  action: PlanParamsChangeAction,
): string => {
  switch (action.type) {
    case 'start':
      return assertFalse()
    case 'startCopiedFromBeforeHistory':
      return assertFalse()
    case 'startCutByClient':
      return assertFalse()
    case 'startFromURL':
      return assertFalse()
    case 'noOpToMarkMigration':
      return `Accepted migration to new inputs`
    // ---------
    // SetMarketDataDay
    // ---------
    case 'setMarketDataDay': {
      const { value } = action
      return `Set market data date to ${CalendarDayFns.toStr(value)}`
    }

    // ---------
    // SetDialogPosition
    // ---------
    case 'setDialogPosition': {
      switch (prevPlanParams.dialogPositionNominal) {
        case 'age':
        case 'current-portfolio-balance':
        case 'future-savings':
        case 'income-during-retirement':
          return `Completed "${planSectionLabel(
            prevPlanParams.dialogPositionNominal,
          )}" section`
        case 'show-results':
          return 'Show Results'
        case 'show-all-inputs':
          return 'Show All Inputs'
        case 'done':
          return assertFalse()
        default:
          return noCase(prevPlanParams.dialogPositionNominal)
      }
    }

    // ---------
    // AddPartner
    // ---------
    case 'addPartner': {
      return `Added partner`
    }

    // ---------
    // DeletePartner
    // ---------
    case 'deletePartner': {
      return `Deleted partner`
    }
    // ---------
    // SetPersonRetired
    // ---------
    case 'setPersonRetired': {
      return `Marked ${
        action.value === 'person1' ? 'yourself' : 'your partner'
      } as retired`
    }

    // ---------
    // SetPersonNotRetired
    // ---------
    case 'setPersonNotRetired': {
      const personType = action.value
      return `Marked ${
        personType === 'person1' ? 'yourself' : 'your partner'
      } as not retired`
    }
    // ---------
    // setPersonCurrentAgeInfo
    // ---------
    case 'setPersonCurrentAgeInfo': {
      const { personId, currentAgeInfo } = action.value
      const prefix = `Set ${yourOrYourPartners(personId)}`
      const postfix = currentAgeInfo.isDatedPlan
        ? `month of birth to ${CalendarMonthFns.toStr(currentAgeInfo.monthOfBirth)}`
        : `current age to ${InMonthsFns.toStr(currentAgeInfo.currentAge)}`
      return `${prefix} ${postfix}`
    }

    // ---------
    // SetPersonRetirementAge
    // ---------
    case 'setPersonRetirementAge': {
      const { person: personType, retirementAge } = action.value
      return `Set ${yourOrYourPartners(
        personType,
      )} retirement age to ${InMonthsFns.toStr(retirementAge)}`
    }

    // ---------
    // SetPersonMaxAge
    // ---------
    case 'setPersonMaxAge': {
      const { person: personType, maxAge } = action.value
      return `Set ${yourOrYourPartners(personType)} max age to ${InMonthsFns.toStr(
        maxAge,
      )}`
    }

    // ---------
    // SetWithdrawalStart
    // ---------
    case 'setWithdrawalStart': {
      const { value } = action
      return `Set withdrawals to start at ${yourOrYourPartners(value)} retirement`
    }

    // -------------- LABELED AMOUNT TIMED/UNTIMED -------------------
    // ---------
    // addLabeledAmountUntimed
    // ---------
    case 'addLabeledAmountUntimed': {
      const { location, entryId, sortIndex } = action.value
      return `Added ${getLabeledAmountUntimedLocationStr(location)} entry `
    }
    // ---------
    // addLabeledAmountTimed2
    // ---------
    case 'addLabeledAmountTimed2': {
      const { location, entryId, amountAndTiming, sortIndex } = action.value
      return `Added ${getLabeledAmountTimedLocationStr(location)} entry`
    }

    // ---------
    // DeleteLabeledAmount
    // ---------
    case 'deleteLabeledAmountTimedOrUntimed': {
      const { location, entryId } = action.value
      const prevEntries: Record<
        string,
        LabeledAmountTimed | LabeledAmountUntimed
      > = PlanParamsHelperFns.getLabeledAmountTimedOrUntimedListFromLocation(
        prevPlanParams,
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
    }
    // ---------
    // setLabelForLabeledAmountTimedOrUntimed
    // ---------
    case 'setLabelForLabeledAmountTimedOrUntimed': {
      const { location, entryId, label } = action.value
      return `Set ${getLabeledAmountTimedOrUntimedLocationStr(
        location,
      )} entry label to "${getLabelStrTruncated(label)}'" `
    }

    // ---------
    // setAmountForLabeledAmountUntimed
    // ---------
    case 'setAmountForLabeledAmountUntimed': {
      const { location, entryId, amount } = action.value
      return `Set ${getLabeledAmountUntimedLocationStr(
        location,
      )} entry "${getLabelStrTruncated(
        fGet(
          optGet(
            PlanParamsHelperFns.getLabeledAmountUntimedListFromLocation(
              planParams,
              location,
            ),
            entryId,
          ),
        ).label,
      )}" amount to ${formatCurrency(amount)} per month`
    }

    // ---------
    // setBaseAmountForLabeledAmountTimed
    // ---------
    case 'setBaseAmountForLabeledAmountTimed': {
      const { location, entryId, baseAmount } = action.value
      const entries = PlanParamsHelperFns.getLabeledAmountTimedListFromLocation(
        planParams,
        location,
      )
      return `Set ${getLabeledAmountTimedLocationStr(
        location,
      )} entry "${getLabelStrTruncated(
        fGet(optGet(entries, entryId)).label,
      )}" amount to ${formatCurrency(baseAmount)} per month`
    }

    // ---------
    // setNominalForLabeledAmountTimedOrUntimed
    // ---------
    case 'setNominalForLabeledAmountTimedOrUntimed': {
      const { location, entryId, nominal } = action.value
      const entries: Record<string, LabeledAmountTimed | LabeledAmountUntimed> =
        PlanParamsHelperFns.getLabeledAmountTimedOrUntimedListFromLocation(
          planParams,
          location,
        )

      return `Set ${getLabeledAmountTimedOrUntimedLocationStr(
        location,
      )} entry "${getLabelStrTruncated(
        fGet(optGet(entries, entryId)).label,
      )}" amount as ${nominal ? 'nominal' : 'rea'} dollars`
    }
    // ---------
    // setMonthRangeForLabeledAmountTimed2
    // ---------
    case 'setMonthRangeForLabeledAmountTimed2': {
      const { location, entryId, monthRange } = action.value
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
    }

    // --------------------------- WEALTH --------------------------------
    // ---------
    // SetCurrentPortfolioBalance
    // ---------
    case 'setCurrentPortfolioBalance': {
      const { value } = action
      return `Set current portfolio balance to ${formatCurrency(value)}`
    }
    // ---------
    // SetSpendingCeiling
    // ---------
    case 'setSpendingCeiling': {
      const { value } = action
      return value === null
        ? 'Removed spending ceiling'
        : `Set spending ceiling to ${formatCurrency(value)}`
    }
    // ---------
    // SetSpendingFloor
    // ---------
    case 'setSpendingFloor': {
      const { value } = action
      return value === null
        ? 'Removed spending floor'
        : `Set spending floor to ${formatCurrency(value)}`
    }

    // ---------
    // SetLegacyTotal
    // ---------
    case 'setLegacyTotal': {
      const { value } = action
      return `Set total legacy target to ${formatCurrency(value)}`
    }
    // --------------------------- ADVANCED --------------------------------
    // ---------
    // SetTPAWRiskTolerance
    // ---------
    case 'setTPAWRiskTolerance': {
      const { value } = action
      return `Set risk tolerance to ${value}`
    }
    // ---------
    // SetTPAWAdditionalSpendingTilt
    // ---------
    case 'setTPAWAdditionalSpendingTilt': {
      const { value } = action
      return `Set extra spending tilt to ${formatPercentage(1)(value)}`
    }

    // ---------
    // SetTPAWRiskDeltaAtMaxAge
    // ---------
    case 'setTPAWRiskDeltaAtMaxAge': {
      const { value } = action
      return `Set decease risk tolerance with age to ${-value}`
    }

    // ---------
    // SetTPAWRiskToleranceForLegacyAsDeltaFromAt20
    // ---------
    case 'setTPAWRiskToleranceForLegacyAsDeltaFromAt20': {
      const { value } = action
      return `Set increase risk tolerance for legacy to ${value}`
    }

    // ---------
    // SetTPAWTimePreference
    // ---------
    case 'setTPAWTimePreference': {
      const { value } = action
      return `Set preference for the future to ${formatPercentage(1)(-value)}`
    }

    // ---------
    // SetSWRWithdrawalAsPercentPerYear
    // ---------
    case 'setSWRWithdrawalAsPercentPerYear': {
      const { value } = action
      return `Set withdrawal rate to ${formatPercentage(1)(value)} per year`
    }

    // ---------
    // SetSWRWithdrawalAsAmountPerMonth
    // ---------
    case 'setSWRWithdrawalAsAmountPerMonth': {
      const { value } = action
      return `Set withdrawal amount to ${formatCurrency(value)} per month`
    }

    // ---------
    // setSPAWAndSWRAllocation2
    // ---------
    case 'setSPAWAndSWRAllocation2': {
      const { value } = action
      return `Updated stock allocation`
    }
    // ---------
    // SetSPAWAnnualSpendingTilt
    // ---------
    case 'setSPAWAnnualSpendingTilt': {
      const { value } = action
      return `Set spending tilt to ${formatPercentage(1)(value)}`
    }

    // ---------
    // SetTPAWAndSPAWLMP
    // ---------
    case 'setTPAWAndSPAWLMP': {
      const { value } = action
      return `Set LMP to ${formatCurrency(value)}`
    }

    // ---------
    // setExpectedReturnsForPlanning
    // ---------
    case 'setExpectedReturnsForPlanning': {
      const { value } = action

      const { forUndoRedo: label } = getExpectedReturnTypeLabelInfo(value)
      const prev =
        prevPlanParams.advanced.returnsStatsForPlanning.expectedValue
          .empiricalAnnualNonLog
      if (prev.type !== value.type) return `Set expected returns to ${label}`

      switch (value.type) {
        case 'regressionPrediction,20YearTIPSYield':
        case 'conservativeEstimate,20YearTIPSYield':
        case '1/CAPE,20YearTIPSYield':
        case 'historical':
          // Can not have consecutive changes of these types because
          // they don't have data.
          return assertFalse()
        case 'fixedEquityPremium':
          return `Set fixed equity premium to ${formatPercentage(1)(value.equityPremium)}`
        case 'custom':
          assert(prev.type === 'custom')
          if (prev.stocks.base !== value.stocks.base) {
            return `Set base for custom expected return of stocks to ${getExpectedReturnCustomStockBaseLabel(value.stocks.base).lowercase}`
          }
          if (prev.stocks.delta !== value.stocks.delta) {
            return `Set delta for custom expected return of stocks to ${formatPercentage(1)(value.stocks.delta)}`
          }
          if (prev.bonds.base !== value.bonds.base) {
            return `Set base for custom expected return of bonds to ${getExpectedReturnCustomBondBaseLabel(value.bonds.base).lowercase}`
          }
          if (prev.bonds.delta !== value.bonds.delta) {
            return `Set delta for custom expected return of bonds fixed delta to ${formatPercentage(1)(value.bonds.delta)}`
          }
          assertFalse()
        case 'fixed':
          assert(prev.type === 'fixed')
          if (value.stocks !== prev.stocks) {
            return `Set expected return of stocks to ${formatPercentage(1)(value.stocks)}`
          }
          if (value.bonds !== prev.bonds) {
            return `Set expected return of bonds to ${formatPercentage(1)(value.bonds)}`
          }
          assertFalse()
        default:
          noCase(value)
      }
    }

    // ---------
    // setReturnsStatsForPlanningStockVolatilityScale
    // ---------
    case 'setReturnsStatsForPlanningStockVolatilityScale': {
      const { value } = action
      return `Set stock volatility scaling to ${value.toFixed(2)}`
    }

    // ---------
    // setHistoricalReturnsAdjustmentBondVolatilityScale
    // ---------
    case 'setHistoricalReturnsAdjustmentBondVolatilityScale': {
      const { value } = action
      return `Set bond volatility scaling to ${value.toFixed(2)}`
    }

    // ---------
    // SetAnnualInflation
    // ---------
    case 'setAnnualInflation': {
      const { value } = action
      return `Set annual inflation to ${
        value.type === 'manual'
          ? `${formatPercentage(1)(value.value)}`
          : _.lowerFirst(inflationTypeLabel(value))
      }`
    }

    // ---------
    // SetSamplingToDefault
    // ---------
    case 'setSamplingToDefault': {
      return `Set simulation to Monte Carlo sequence`
    }

    // ---------
    // SetSampling
    // ---------
    case 'setSampling': {
      const { value } = action
      return `Set simulation to ${
        value === 'historical' ? 'historical' : ' Monte Carlo'
      } sequence`
    }

    // ---------
    // SetMonteCarloSamplingBlockSize2
    // ---------
    case 'setMonteCarloSamplingBlockSize2': {
      const { value } = action
      return `Set block size for Monte Carlo simulation to ${InMonthsFns.toStr(value)}`
    }

    // ---------
    // SetMonteCarloSamplingStaggerRunStarts
    // ---------
    case 'setMonteCarloStaggerRunStarts': {
      const { value } = action
      return `Set stagger run starts for Monte Carlo simulation to ${value ? 'true' : 'false'}`
    }

    // ---------
    // SetStrategy
    // ---------
    case 'setStrategy': {
      const { value } = action
      return `Set strategy to ${value}`
    }

    // ---------
    // setHistoricalMonthlyLogReturnsAdjustmentOverrideToFixedForTesting2
    // ---------
    case 'setHistoricalMonthlyLogReturnsAdjustmentOverrideToFixedForTesting2': {
      const { value } = action
      const valueStr = block(() => {
        switch (value.type) {
          case 'none':
            return 'No Override'
          case 'useExpectedReturnsForPlanning':
            return 'Expected Returns'
          case 'manual':
            return `Manual: (stocks: ${formatPercentage(1)(value.stocks)}, bonds: ${formatPercentage(1)(value.bonds)})`
        }
      })
      return `Set historical returns adjustment override to fixed for testing: ${valueStr}`
    }

    default:
      return _renderPlanParamsChangeActionDeprecated(
        prevPlanParams,
        planParams,
        action,
      )
  }
}

// -------------------------------------------------------------------------
// DEPRECATED
// -------------------------------------------------------------------------
const _renderPlanParamsChangeActionDeprecated = (
  prevPlanParams: PlanParams,
  planParams: PlanParams,
  action: PlanParamsChangeActionDeprecated,
): string => {
  switch (action.type) {
    case 'switchHistoricalReturns': {
      const { value } = action
      const stocksAndBondsStr = ({
        stocks,
        bonds,
      }: {
        stocks: number
        bonds: number
      }) =>
        `stocks: ${formatPercentage(1)(stocks)}, bonds: ${formatPercentage(1)(
          bonds,
        )}`
      return `Switch historical returns to ${
        value.type === 'adjusted'
          ? value.adjustment.type === 'toExpected'
            ? 'adjusted to expected'
            : value.adjustment.type === 'to' || value.adjustment.type === 'by'
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
    }

    case 'setHistoricalReturnsAdjustForBlockSampling': {
      const { value } = action
      return `${value ? 'Set' : 'Unset'} correct historical returns for block sampling.`
    }

    case 'setHistoricalReturnsBonds': {
      const { value } = action
      return value === 'adjustExpectedToExpectedUsedForPlanning'
        ? 'Bond volatility enabled'
        : value === 'fixedToExpectedUsedForPlanning'
          ? 'Bond volatility disabled'
          : noCase(value)
    }

    case 'setHistoricalReturnsFixedStocks': {
      const { value } = action
      return `Set fixed historical returns stocks to ${formatPercentage(1)(value)}`
    }

    case 'setHistoricalReturnsFixedBonds': {
      const { value } = action
      return `Set fixed historical returns bonds to ${formatPercentage(1)(value)}`
    }

    case 'setHistoricalReturnsStocksDev': {
      return `DEV: Set historical returns for stocks`
    }

    case 'setHistoricalReturnsBondsDev': {
      return `DEV: Set historical returns for bonds`
    }

    case 'setExpectedReturns': {
      const { value } = action
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
    }

    case 'setHistoricalReturnsAdjustExpectedReturnDev': {
      const { type } = action.value
      return `DEV: Set historical returns expected value adjustment for ${
        type === 'stocks' ? 'stocks' : 'bonds'
      }`
    }

    case 'addValueForMonthRange': {
      const { location } = action.value
      return `Added ${getLabeledAmountTimedLocationStr(location)} entry`
    }

    case 'deleteLabeledAmount': {
      const { location, entryId } = action.value
      const prevEntries: Record<
        string,
        LabeledAmountTimed | LabeledAmountUntimed
      > = PlanParamsHelperFns.getLabeledAmountTimedOrUntimedListFromLocation(
        prevPlanParams,
        location,
      )
      const prevEntry = optGet(prevEntries, entryId)
      const label = prevEntry ? getLabelStrTruncated(prevEntry.label) : ''
      return `Deleted ${getLabeledAmountTimedOrUntimedLocationStr(
        location,
      )} entry ${label}`
    }

    case 'addLabeledAmount': {
      const { location } = action.value
      return `Added ${getLabeledAmountTimedOrUntimedLocationStr(location)} entry `
    }

    case 'setLabelForLabeledAmount': {
      const { location, label } = action.value
      return `Set ${getLabeledAmountTimedOrUntimedLocationStr(
        location,
      )} entry label to "${getLabelStrTruncated(label)}'" `
    }

    case 'setAmountForLabeledAmount': {
      const { location, entryId, amount } = action.value
      const entries: Record<string, LabeledAmountUntimed | LabeledAmountTimed> =
        PlanParamsHelperFns.getLabeledAmountTimedOrUntimedListFromLocation(
          planParams,
          location,
        )
      return `Set ${getLabeledAmountTimedOrUntimedLocationStr(
        location,
      )} entry "${getLabelStrTruncated(
        fGet(optGet(entries, entryId)).label,
      )}" amount to ${formatCurrency(amount)} per month`
    }

    case 'setNominalForLabeledAmount': {
      const { location, entryId, nominal } = action.value
      const entries: Record<string, LabeledAmountTimed | LabeledAmountUntimed> =
        PlanParamsHelperFns.getLabeledAmountTimedOrUntimedListFromLocation(
          planParams,
          location,
        )

      return `Set ${getLabeledAmountTimedOrUntimedLocationStr(
        location,
      )} entry "${getLabelStrTruncated(
        fGet(optGet(entries, entryId)).label,
      )}" amount as ${nominal ? 'nominal' : 'rea'} dollars`
    }

    case 'setMonthRangeForValueForMonthRange': {
      const { location, entryId } = action.value
      const entries: Record<string, LabeledAmountTimed | LabeledAmountUntimed> =
        PlanParamsHelperFns.getLabeledAmountTimedListFromLocation(
          planParams,
          location,
        )
      return `Set ${getLabeledAmountTimedLocationStr(
        location,
      )} entry "${getLabelStrTruncated(
        fGet(optGet(entries, entryId)).label,
      )}" month range`
    }

    case 'setPersonMonthOfBirth': {
      const { person, monthOfBirth } = action.value
      return `Set ${yourOrYourPartners(
        person,
      )} month of birth to ${CalendarMonthFns.toStr(monthOfBirth)}`
    }

    case 'setMonteCarloSamplingBlockSize': {
      const { value } = action
      return `Set block size for Monte Carlo simulation to ${InMonthsFns.toStr({ inMonths: value })}`
    }

    case 'setHistoricalStockReturnsAdjustmentVolatilityScale': {
      const { value } = action
      return `Set stocks volatility scaling to ${value}`
    }

    case 'setHistoricalBondReturnsAdjustmentEnableVolatility': {
      const { value } = action
      return `Bond volatility ${value ? 'enabled' : 'disabled'}`
    }

    case 'setExpectedReturns2': {
      const { value } = action
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
    }

    case 'setPersonMonthOfBirth2': {
      const { person, monthOfBirth } = action.value
      return `Set ${yourOrYourPartners(
        person,
      )} month of birth to ${CalendarMonthFns.toStr(monthOfBirth)}`
    }

    case 'setSPAWAndSWRAllocation': {
      return `Updated stock allocation`
    }

    case 'setMonthRangeForLabeledAmountTimed': {
      const { location, entryId } = action.value
      const entries: Record<string, LabeledAmountTimed | LabeledAmountUntimed> =
        PlanParamsHelperFns.getLabeledAmountTimedListFromLocation(
          planParams,
          location,
        )
      return `Set ${getLabeledAmountTimedLocationStr(
        location,
      )} entry "${getLabelStrTruncated(
        fGet(optGet(entries, entryId)).label,
      )}" month range`
    }

    case 'addLabeledAmountTimed': {
      const { location } = action.value
      return `Added ${getLabeledAmountTimedLocationStr(location)} entry`
    }

    case 'setHistoricalMonthlyLogReturnsAdjustmentOverrideToFixedForTesting': {
      const { value } = action
      return `Set historical monthly log returns adjustment override to fixed for testing to ${value}`
    }

    default:
      return noCase(action)
  }
}

export const getLabeledAmountTimedLocationStr = (
  location: LabeledAmountTimedLocation,
) => {
  switch (location) {
    case 'extraSpendingEssential':
      return 'essential expense'
    case 'extraSpendingDiscretionary':
      return 'discretionary expense'
    case 'futureSavings':
      return _.lowerCase(planSectionLabel('future-savings'))
    case 'incomeDuringRetirement':
      return _.lowerCase(planSectionLabel('income-during-retirement'))
    default:
      noCase(location)
  }
}

export const getLabeledAmountUntimedLocationStr = (
  location: LabeledAmountUntimedLocation,
) => {
  switch (location) {
    case 'legacyExternalSources':
      return 'legacy non-portfolio source'
    default:
      noCase(location)
  }
}
export const getLabeledAmountTimedOrUntimedLocationStr = (
  location: LabeledAmountTimedOrUntimedLocation,
) => {
  switch (location) {
    case 'legacyExternalSources':
      return getLabeledAmountUntimedLocationStr(location)
    default:
      return getLabeledAmountTimedLocationStr(location)
  }
}

export const getLabelStrTruncated = (label: string | null) =>
  label ? _.truncate(label, { length: 30 }) : '<no label>'
