import * as Sentry from '@sentry/nextjs'
import {
  CalendarMonthFns,
  DEFAULT_ANNUAL_SWR_WITHDRAWAL_PERCENT,
  LabeledAmountUntimed,
  LabeledAmountUntimedLocation,
  PlanParams,
  PlanParamsChangeAction,
  PlanParamsChangeActionCurrent,
  LabeledAmountTimed,
  LabeledAmountTimedLocation,
  assert,
  assertFalse,
  fGet,
  noCase,
  LabeledAmountTimedOrUntimedLocation,
  block,
  partialDefaultDatelessPlanParams,
} from '@tpaw/common'
import _ from 'lodash'
import { PlanParamsNormalized } from '../../../../Simulator/NormalizePlanParams/NormalizePlanParams'
import { PlanParamsHelperFns } from '../../../../Simulator/PlanParamsHelperFns'
import { formatCurrency } from '../../../../Utils/FormatCurrency'
import { formatPercentage } from '../../../../Utils/FormatPercentage'
import { yourOrYourPartners } from '../../../../Utils/YourOrYourPartners'
import { optGet } from '../../../../Utils/optGet'
import { planSectionLabel } from '../../Plan/PlanInput/Helpers/PlanSectionLabel'
import {} from '../../Plan/PlanInput/PlanInputExpectedReturnsAndVolatility'
import { getDeletePartnerChangeActionImpl } from './GetDeletePartnerChangeActionImpl'
import { getSetPersonRetiredChangeActionImpl } from './GetSetPersonRetiredChangeActionImpl'
import { InMonthsFns } from '../../../../Utils/InMonthsFns'
import {
  getExpectedReturnTypeLabelInfo,
  getExpectedReturnCustomStockBaseLabel,
  getExpectedReturnCustomBondBaseLabel,
} from '../../Plan/PlanInput/PlanInputExpectedReturnsAndVolatilityFns'
import { CalendarDayFns } from '../../../../Utils/CalendarDayFns'
import { inflationTypeLabel } from '../../Plan/PlanInput/PlanInputInflationFns'

const { getPerson } = PlanParamsHelperFns
export type PlanParamsChangeActionImpl = {
  applyToClone: (
    clone: PlanParams,
    planParamsNorm: PlanParamsNormalized,
  ) => void | PlanParams // If a value is returned, it supercedes the clone.
  render: (
    prevPlanParams: PlanParams,
    planParams: PlanParams,
  ) => React.ReactNode
  merge: false | true | ((prev: PlanParamsChangeAction) => boolean)
}

export const getPlanParamsChangeActionImpl = (
  action: PlanParamsChangeActionCurrent,
): PlanParamsChangeActionImpl => {
  switch (action.type) {
    case 'start':
      return {
        applyToClone: () => assertFalse(),
        render: () => assertFalse(),
        merge: false,
      }
    case 'startCopiedFromBeforeHistory':
      return {
        applyToClone: () => assertFalse(),
        render: () => assertFalse(),
        merge: false,
      }
    case 'startCutByClient':
      return {
        applyToClone: () => assertFalse(),
        render: () => assertFalse(),
        merge: false,
      }
    case 'startFromURL':
      return {
        applyToClone: () => assertFalse(),
        render: () => assertFalse(),
        merge: false,
      }
    case 'noOpToMarkMigration':
      return {
        applyToClone: () => {}, // Do nothing.
        render: () => `Accepted migration to new inputs`,
        merge: false,
      }
    // ---------
    // SetMarketDataDay
    // ---------
    case 'setMarketDataDay': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          assert(!clone.datingInfo.isDated)
          clone.datingInfo.marketDataAsOfEndOfDayInNY = value
        },
        render: () => `Set market data date to ${CalendarDayFns.toStr(value)}`,
        merge: false,
      }
    }

    // ---------
    // SetDialogPosition
    // ---------
    case 'setDialogPosition': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.dialogPositionNominal = value
        },
        render: (prevParams) => {
          switch (prevParams.dialogPositionNominal) {
            case 'age':
            case 'current-portfolio-balance':
            case 'future-savings':
            case 'income-during-retirement':
              return `Completed "${planSectionLabel(
                prevParams.dialogPositionNominal,
              )}" section`
            case 'show-results':
              return 'Show Results'
            case 'show-all-inputs':
              return 'Show All Inputs'
            case 'done':
              return assertFalse()
            default:
              noCase(prevParams.dialogPositionNominal)
          }
        },
        merge: false,
      }
    }

    // ---------
    // AddPartner
    // ---------
    case 'addPartner': {
      return {
        applyToClone: (clone) => {
          assert(!clone.people.withPartner)
          clone.people = {
            withPartner: true,
            person1: clone.people.person1,
            person2: _.cloneDeep(clone.people.person1),
            withdrawalStart: 'person1',
          }
        },
        render: () => `Added partner`,
        merge: false,
      }
    }

    // ---------
    // DeletePartner
    // ---------
    case 'deletePartner': {
      return getDeletePartnerChangeActionImpl()
    }
    // ---------
    // SetPersonRetired
    // ---------
    case 'setPersonRetired':
      return getSetPersonRetiredChangeActionImpl(action)

    // ---------
    // SetPersonNotRetired
    // ---------
    case 'setPersonNotRetired': {
      const personType = action.value
      return {
        applyToClone: (clone, planParamsNorm) => {
          const currPerson = getPerson(clone, personType)
          const defaultPerson = _.cloneDeep(
            partialDefaultDatelessPlanParams.people.person1,
          )
          assert(defaultPerson.ages.type === 'retirementDateSpecified')
          const retirementAge = {
            inMonths: _.clamp(
              defaultPerson.ages.retirementAge.inMonths,
              fGet(planParamsNorm.ages[personType]).currentAgeInfo.inMonths + 1,
              currPerson.ages.maxAge.inMonths - 1,
            ),
          }
          currPerson.ages = {
            type: 'retirementDateSpecified',
            currentAgeInfo: currPerson.ages.currentAgeInfo,
            maxAge: currPerson.ages.maxAge,
            retirementAge,
          }
        },
        render: () =>
          `Marked ${
            personType === 'person1' ? 'yourself' : 'your partner'
          } as not retired`,
        merge: false,
      }
    }
    // ---------
    // setPersonCurrentAgeInfo
    // ---------
    case 'setPersonCurrentAgeInfo': {
      const { personId, currentAgeInfo } = action.value
      return {
        applyToClone: (clone) => {
          getPerson(clone, personId).ages.currentAgeInfo = currentAgeInfo
        },
        render: () => {
          const prefix = `Set ${yourOrYourPartners(personId)}`
          const postfix = currentAgeInfo.isDatedPlan
            ? `month of birth to ${CalendarMonthFns.toStr(currentAgeInfo.monthOfBirth)}`
            : `current age to ${InMonthsFns.toStr(currentAgeInfo.currentAge)}`
          return `${prefix} ${postfix}`
        },
        merge: (prev) =>
          prev.type === 'setPersonCurrentAgeInfo' &&
          prev.value.personId === personId,
      }
    }

    // ---------
    // SetPersonRetirementAge
    // ---------
    case 'setPersonRetirementAge': {
      const { person: personType, retirementAge } = action.value
      return {
        applyToClone: (clone) => {
          const person = getPerson(clone, personType)
          assert(person.ages.type === 'retirementDateSpecified')
          person.ages.retirementAge = retirementAge
        },
        render: () =>
          `Set ${yourOrYourPartners(
            personType,
          )} retirement age to ${InMonthsFns.toStr(retirementAge)}`,
        merge: (prev) =>
          prev.type === 'setPersonRetirementAge' &&
          prev.value.person === personType,
      }
    }

    // ---------
    // SetPersonMaxAge
    // ---------
    case 'setPersonMaxAge': {
      const { person: personType, maxAge } = action.value
      return {
        applyToClone: (clone) => {
          const person = getPerson(clone, personType)
          person.ages.maxAge = maxAge
        },
        render: () =>
          `Set ${yourOrYourPartners(personType)} max age to ${InMonthsFns.toStr(
            maxAge,
          )}`,
        merge: (prev) =>
          prev.type === 'setPersonMaxAge' && prev.value.person === personType,
      }
    }

    // ---------
    // SetWithdrawalStart
    // ---------
    case 'setWithdrawalStart': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          assert(clone.people.withPartner)
          clone.people.withdrawalStart = value
        },
        render: () =>
          `Set withdrawals to start at ${yourOrYourPartners(value)} retirement`,
        merge: false,
      }
    }

    // -------------- LABELED AMOUNT TIMED/UNTIMED -------------------
    // ---------
    // addLabeledAmountUntimed
    // ---------
    case 'addLabeledAmountUntimed': {
      const { location, entryId, sortIndex } = action.value
      return {
        applyToClone: (clone) => {
          const entries =
            PlanParamsHelperFns.getLabeledAmountUntimedListFromLocation(
              clone,
              location,
            )
          assert(!optGet(entries, entryId))
          entries[entryId] = {
            id: entryId,
            label: null,
            amount: 0,
            nominal: false,
            sortIndex,
            colorIndex: _getNextColorIndex(
              _getUsedColorIndexesByLocation(clone, location),
            ),
          }
        },
        render: () => {
          return `Added ${getLabeledAmountUntimedLocationStr(location)} entry `
        },
        merge: false,
      }
    }
    // ---------
    // addLabeledAmountTimed2
    // ---------
    case 'addLabeledAmountTimed2': {
      const { location, entryId, amountAndTiming, sortIndex } = action.value
      return {
        applyToClone: (clone) => {
          const values =
            PlanParamsHelperFns.getLabeledAmountTimedListFromLocation(
              clone,
              location,
            )
          assert(!optGet(values, entryId))

          values[entryId] = {
            id: entryId,
            sortIndex,
            label: null,
            nominal: false,
            colorIndex: _getNextColorIndex(
              _getUsedColorIndexesByLocation(clone, location),
            ),
            amountAndTiming,
          }
        },
        render: () => {
          return `Added ${getLabeledAmountTimedLocationStr(location)} entry`
        },
        merge: false,
      }
    }

    // ---------
    // DeleteLabeledAmount
    // ---------
    case 'deleteLabeledAmountTimedOrUntimed': {
      const { location, entryId } = action.value
      return {
        applyToClone: (clone) => {
          const entries: Record<
            string,
            LabeledAmountTimed | LabeledAmountUntimed
          > =
            PlanParamsHelperFns.getLabeledAmountTimedOrUntimedListFromLocation(
              clone,
              location,
            )
          assert(optGet(entries, entryId))
          delete entries[entryId]
        },
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
        merge: false,
      }
    }
    // ---------
    // setLabelForLabeledAmountTimedOrUntimed
    // ---------
    case 'setLabelForLabeledAmountTimedOrUntimed': {
      const { location, entryId, label } = action.value
      return {
        applyToClone: (clone) => {
          const entries: Record<
            string,
            LabeledAmountTimed | LabeledAmountUntimed
          > =
            PlanParamsHelperFns.getLabeledAmountTimedOrUntimedListFromLocation(
              clone,
              location,
            )
          fGet(optGet(entries, entryId)).label = label
        },
        render: () => {
          return `Set ${getLabeledAmountTimedOrUntimedLocationStr(
            location,
          )} entry label to "${getLabelStrTruncated(label)}'" `
        },
        merge: (prev) =>
          prev.type === 'setLabelForLabeledAmount' &&
          prev.value.entryId === entryId,
      }
    }

    // ---------
    // setAmountForLabeledAmountUntimed
    // ---------
    case 'setAmountForLabeledAmountUntimed': {
      const { location, entryId, amount } = action.value
      return {
        applyToClone: (clone) => {
          fGet(
            optGet(
              PlanParamsHelperFns.getLabeledAmountUntimedListFromLocation(
                clone,
                location,
              ),
              entryId,
            ),
          ).amount = amount
        },
        render: (_: PlanParams, planParams: PlanParams) => {
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
        },

        merge: (prev) =>
          prev.type === 'setAmountForLabeledAmount' &&
          prev.value.entryId === entryId,
      }
    }

    // ---------
    // setBaseAmountForLabeledAmountTimed
    // ---------
    case 'setBaseAmountForLabeledAmountTimed': {
      const { location, entryId, baseAmount } = action.value
      return {
        applyToClone: (clone) => {
          const entry = fGet(
            optGet(
              PlanParamsHelperFns.getLabeledAmountTimedListFromLocation(
                clone,
                location,
              ),
              entryId,
            ),
          )
          assert(entry.amountAndTiming.type === 'recurring')
          entry.amountAndTiming.baseAmount = baseAmount
        },
        render: (_: PlanParams, planParams: PlanParams) => {
          const entries =
            PlanParamsHelperFns.getLabeledAmountTimedListFromLocation(
              planParams,
              location,
            )
          return `Set ${getLabeledAmountTimedLocationStr(
            location,
          )} entry "${getLabelStrTruncated(
            fGet(optGet(entries, entryId)).label,
          )}" amount to ${formatCurrency(baseAmount)} per month`
        },

        merge: (prev) =>
          prev.type === 'setAmountForLabeledAmount' &&
          prev.value.entryId === entryId,
      }
    }

    // ---------
    // setNominalForLabeledAmountTimedOrUntimed
    // ---------
    case 'setNominalForLabeledAmountTimedOrUntimed': {
      const { location, entryId, nominal } = action.value
      return {
        applyToClone: (clone) => {
          const entries: Record<
            string,
            LabeledAmountTimed | LabeledAmountUntimed
          > =
            PlanParamsHelperFns.getLabeledAmountTimedOrUntimedListFromLocation(
              clone,
              location,
            )
          fGet(optGet(entries, entryId)).nominal = nominal
        },
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
        merge: false,
      }
    }
    // ---------
    // setMonthRangeForLabeledAmountTimed2
    // ---------
    case 'setMonthRangeForLabeledAmountTimed2': {
      const { location, entryId, monthRange } = action.value
      return {
        applyToClone: (clone) => {
          const entry = fGet(
            optGet(
              PlanParamsHelperFns.getLabeledAmountTimedListFromLocation(
                clone,
                location,
              ),
              entryId,
            ),
          )
          assert(entry.amountAndTiming.type !== 'oneTime')
          assert(entry.amountAndTiming.type !== 'inThePast')
          entry.amountAndTiming.monthRange = monthRange
        },
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
        merge: (prev) =>
          prev.type === 'setMonthRangeForLabeledAmountTimed' &&
          prev.value.entryId === entryId,
      }
    }

    // --------------------------- WEALTH --------------------------------
    // ---------
    // SetCurrentPortfolioBalance
    // ---------
    case 'setCurrentPortfolioBalance': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.wealth.portfolioBalance = clone.datingInfo.isDated
            ? {
                isDatedPlan: true,
                updatedHere: true,
                amount: value,
              }
            : {
                isDatedPlan: false,
                amount: value,
              }
        },
        render: () =>
          `Set current portfolio balance to ${formatCurrency(value)}`,
        merge: () => true,
      }
    }
    // ---------
    // SetSpendingCeiling
    // ---------
    case 'setSpendingCeiling': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.adjustmentsToSpending.tpawAndSPAW.monthlySpendingCeiling = value
        },
        render: () =>
          value === null
            ? 'Removed spending ceiling'
            : `Set spending ceiling to ${formatCurrency(value)}`,
        merge: (prev) =>
          prev.type === 'setSpendingCeiling' &&
          prev.value !== null &&
          value !== null,
      }
    }
    // ---------
    // SetSpendingFloor
    // ---------
    case 'setSpendingFloor': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.adjustmentsToSpending.tpawAndSPAW.monthlySpendingFloor = value
        },
        render: () =>
          value === null
            ? 'Removed spending floor'
            : `Set spending floor to ${formatCurrency(value)}`,
        merge: (prev) =>
          prev.type === 'setSpendingFloor' &&
          prev.value !== null &&
          value !== null,
      }
    }

    // ---------
    // SetLegacyTotal
    // ---------
    case 'setLegacyTotal': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.adjustmentsToSpending.tpawAndSPAW.legacy.total = value
        },
        render: () => `Set total legacy target to ${formatCurrency(value)}`,
        merge: () => true,
      }
    }
    // --------------------------- ADVANCED --------------------------------
    // ---------
    // SetTPAWRiskTolerance
    // ---------
    case 'setTPAWRiskTolerance': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.risk.tpaw.riskTolerance.at20 = value
        },
        render: () => `Set risk tolerance to ${value}`,
        merge: () => true,
      }
    }
    // ---------
    // SetTPAWAdditionalSpendingTilt
    // ---------
    case 'setTPAWAdditionalSpendingTilt': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.risk.tpaw.additionalAnnualSpendingTilt = value
        },
        render: () =>
          `Set extra spending tilt to ${formatPercentage(1)(value)}`,
        merge: () => true,
      }
    }

    // ---------
    // SetTPAWRiskDeltaAtMaxAge
    // ---------
    case 'setTPAWRiskDeltaAtMaxAge': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.risk.tpaw.riskTolerance.deltaAtMaxAge = value
        },
        render: () => `Set decease risk tolerance with age to ${-value}`,
        merge: () => true,
      }
    }

    // ---------
    // SetTPAWRiskToleranceForLegacyAsDeltaFromAt20
    // ---------
    case 'setTPAWRiskToleranceForLegacyAsDeltaFromAt20': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.risk.tpaw.riskTolerance.forLegacyAsDeltaFromAt20 = value
        },
        render: () => `Set increase risk tolerance for legacy to ${value}`,
        merge: () => true,
      }
    }

    // ---------
    // SetTPAWTimePreference
    // ---------
    case 'setTPAWTimePreference': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.risk.tpaw.timePreference = value
        },
        render: () =>
          `Set preference for the future to ${formatPercentage(1)(-value)}`,
        merge: () => true,
      }
    }

    // ---------
    // SetSWRWithdrawalAsPercentPerYear
    // ---------
    case 'setSWRWithdrawalAsPercentPerYear': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.risk.swr.withdrawal = {
            type: 'asPercentPerYear',
            percentPerYear: value,
          }
        },
        render: () =>
          `Set withdrawal rate to ${formatPercentage(1)(value)} per year`,
        merge: () => true,
      }
    }

    // ---------
    // SetSWRWithdrawalAsAmountPerMonth
    // ---------
    case 'setSWRWithdrawalAsAmountPerMonth': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.risk.swr.withdrawal = {
            type: 'asAmountPerMonth',
            amountPerMonth: value,
          }
        },
        render: () =>
          `Set withdrawal amount to ${formatCurrency(value)} per month`,
        merge: () => true,
      }
    }

    // ---------
    // setSPAWAndSWRAllocation2
    // ---------
    case 'setSPAWAndSWRAllocation2': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.risk.spawAndSWR.allocation = value
        },
        render: () => `Updated stock allocation`,
        merge: () => true,
      }
    }
    // ---------
    // SetSPAWAnnualSpendingTilt
    // ---------
    case 'setSPAWAnnualSpendingTilt': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.risk.spaw.annualSpendingTilt = value
        },
        render: () => `Set spending tilt to ${formatPercentage(1)(value)}`,
        merge: () => true,
      }
    }

    // ---------
    // SetTPAWAndSPAWLMP
    // ---------
    case 'setTPAWAndSPAWLMP': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.risk.tpawAndSPAW.lmp = value
        },

        render: () => `Set LMP to ${formatCurrency(value)}`,
        merge: () => true,
      }
    }

    // ---------
    // setExpectedReturnsForPlanning
    // ---------
    case 'setExpectedReturnsForPlanning': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.advanced.returnsStatsForPlanning.expectedValue.empiricalAnnualNonLog =
            value
        },
        render: (prevPlanParams) => {
          const { forUndoRedo: label } = getExpectedReturnTypeLabelInfo(value)
          const prev =
            prevPlanParams.advanced.returnsStatsForPlanning.expectedValue
              .empiricalAnnualNonLog
          if (prev.type !== value.type)
            return `Set expected returns to ${label}`

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
        },
        merge: (prev) =>
          prev.type === 'setExpectedReturns' && prev.value.type === value.type,
      }
    }

    // ---------
    // setReturnsStatsForPlanningStockVolatilityScale
    // ---------
    case 'setReturnsStatsForPlanningStockVolatilityScale': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.advanced.returnsStatsForPlanning.standardDeviation.stocks.scale.log =
            value
        },
        render: () => `Set stock volatility scaling to ${value.toFixed(2)}`,
        merge: true,
      }
    }

    // ---------
    // setHistoricalReturnsAdjustmentBondVolatilityScale
    // ---------
    case 'setHistoricalReturnsAdjustmentBondVolatilityScale': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.advanced.historicalReturnsAdjustment.standardDeviation.bonds.scale.log =
            value
        },
        render: () => `Set bond volatility scaling to ${value.toFixed(2)}`,
        merge: true,
      }
    }

    // ---------
    // SetAnnualInflation
    // ---------
    case 'setAnnualInflation': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.advanced.annualInflation = value
        },
        render: () =>
          `Set annual inflation to ${
            value.type === 'manual'
              ? `${formatPercentage(1)(value.value)}`
              : _.lowerFirst(inflationTypeLabel(value))
          }`,
        merge: (prev) =>
          prev.type === 'setAnnualInflation' &&
          prev.value.type === 'manual' &&
          value.type === 'manual',
      }
    }

    // ---------
    // SetSamplingToDefault
    // ---------
    case 'setSamplingToDefault': {
      return {
        applyToClone: (clone) => {
          const defaultSampling = _.clone(
            partialDefaultDatelessPlanParams.advanced.sampling,
          )
          assert(defaultSampling.type === 'monteCarlo')
          // Recreating this object instead of using defaultSampling makes sure we
          // are adding default data correctly. There is none currently , but if
          // gets added it will lead to a compilation error here.
          clone.advanced.sampling = {
            type: defaultSampling.type,
            data: {
              blockSize: { inMonths: defaultSampling.data.blockSize.inMonths },
              staggerRunStarts: defaultSampling.data.staggerRunStarts,
            },
          }
        },
        render: (_: PlanParams, planParams: PlanParams) =>
          `Set simulation to Monte Carlo sequence`,
        merge: false,
      }
    }

    // ---------
    // SetSampling
    // ---------
    case 'setSampling': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          switch (value) {
            case 'monteCarlo':
              if (clone.advanced.sampling.type === 'monteCarlo') return
              const defaultSampling = _.cloneDeep(
                partialDefaultDatelessPlanParams.advanced.sampling,
              )
              assert(defaultSampling.type === 'monteCarlo')
              clone.advanced.sampling = {
                type: 'monteCarlo',
                data:
                  clone.advanced.sampling.defaultData.monteCarlo ??
                  defaultSampling.data,
              }
              break
            case 'historical':
              if (clone.advanced.sampling.type === 'historical') return
              clone.advanced.sampling = {
                type: 'historical',
                defaultData: { monteCarlo: clone.advanced.sampling.data },
              }
              break
            default:
              noCase(value)
          }
        },
        render: () =>
          `Set simulation to ${
            value === 'historical' ? 'historical' : ' Monte Carlo'
          } sequence`,
        merge: false,
      }
    }

    // ---------
    // SetMonteCarloSamplingBlockSize2
    // ---------
    case 'setMonteCarloSamplingBlockSize2': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          assert(clone.advanced.sampling.type === 'monteCarlo')
          clone.advanced.sampling.data.blockSize = value
        },
        render: () =>
          `Set block size for Monte Carlo simulation to ${InMonthsFns.toStr(value)}`,
        merge: () => true,
      }
    }

    // ---------
    // SetMonteCarloSamplingStaggerRunStarts
    // ---------
    case 'setMonteCarloStaggerRunStarts': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          assert(clone.advanced.sampling.type === 'monteCarlo')
          clone.advanced.sampling.data.staggerRunStarts = value
        },
        render: () =>
          `Set stagger run starts for Monte Carlo simulation to ${value ? 'true' : 'false'}`,
        merge: false,
      }
    }

    // ---------
    // SetStrategy
    // ---------
    case 'setStrategy': {
      const { value } = action
      return {
        applyToClone: (clone, planParamsNorm) => {
          clone.advanced.strategy = value
          if (value === 'SWR' && clone.risk.swr.withdrawal.type === 'default') {
            clone.risk.swr.withdrawal = {
              type: 'asPercentPerYear',
              percentPerYear: DEFAULT_ANNUAL_SWR_WITHDRAWAL_PERCENT(
                planParamsNorm.ages.simulationMonths.numWithdrawalMonths,
              ),
            }
          }
        },
        render: () => `Set strategy to ${value}`,
        merge: false,
      }
    }

    // ---------
    // setHistoricalMonthlyLogReturnsAdjustmentOverrideToFixedForTesting2
    // ---------
    case 'setHistoricalMonthlyLogReturnsAdjustmentOverrideToFixedForTesting2': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.advanced.historicalReturnsAdjustment.overrideToFixedForTesting =
            value
        },
        render: () => {
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
        },
        merge: false,
      }
    }
  }
}

const _getUsedColorIndexesByLocation = (
  planParams: PlanParams,
  location: LabeledAmountUntimedLocation | LabeledAmountTimedLocation,
): Set<number> => {
  switch (location) {
    case 'extraSpendingEssential':
    case 'extraSpendingDiscretionary':
      return new Set(
        [
          ..._.values(planParams.adjustmentsToSpending.extraSpending.essential),
          ..._.values(
            planParams.adjustmentsToSpending.extraSpending.discretionary,
          ),
        ].map((x) => x.colorIndex),
      )
    case 'futureSavings':
    case 'incomeDuringRetirement':
      return new Set(
        [
          ..._.values(planParams.wealth.futureSavings),
          ..._.values(planParams.wealth.incomeDuringRetirement),
        ].map((x) => x.colorIndex),
      )
    case 'legacyExternalSources':
      return new Set(
        [
          ..._.values(
            planParams.adjustmentsToSpending.tpawAndSPAW.legacy.external,
          ),
        ].map((x) => x.colorIndex),
      )
    default:
      noCase(location)
  }
}

const _getNextColorIndex = (usedColorIndexes: Set<number>) => {
  let colorIndex = 0
  while (true) {
    if (!usedColorIndexes.has(colorIndex)) return colorIndex
    colorIndex++
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
