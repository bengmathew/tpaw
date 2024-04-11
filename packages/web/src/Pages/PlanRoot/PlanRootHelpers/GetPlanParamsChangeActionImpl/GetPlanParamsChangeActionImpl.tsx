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
} from '@tpaw/common'
import _ from 'lodash'
import { PlanParamsNormalized } from '../../../../UseSimulator/NormalizePlanParams/NormalizePlanParams'
import { PlanParamsHelperFns } from '../../../../UseSimulator/PlanParamsHelperFns'
import { formatCurrency } from '../../../../Utils/FormatCurrency'
import { formatPercentage } from '../../../../Utils/FormatPercentage'
import { yourOrYourPartners } from '../../../../Utils/YourOrYourPartners'
import { optGet } from '../../../../Utils/optGet'
import { planSectionLabel } from '../../Plan/PlanInput/Helpers/PlanSectionLabel'
import { expectedReturnTypeLabelInfo } from '../../Plan/PlanInput/PlanInputExpectedReturnsAndVolatility'
import { inflationTypeLabel } from '../../Plan/PlanInput/PlanInputInflation'
import { getDeletePartnerChangeActionImpl } from './GetDeletePartnerChangeActionImpl'
import { getSetPersonRetiredChangeActionImpl } from './GetSetPersonRetiredChangeActionImpl'
import { InMonthsFns } from '../../../../Utils/InMonthsFns'

const { getPerson } = PlanParamsHelperFns
export type PlanParamsChangeActionImpl = {
  applyToClone: (
    clone: PlanParams,
    planParamsNorm: PlanParamsNormalized,
    defaultPlanParams: PlanParams,
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
        applyToClone: (clone, planParamsNorm, defaultPlanParams) => {
          const currPerson = getPerson(clone, personType)
          const defaultPerson = defaultPlanParams.people.person1
          assert(defaultPerson.ages.type === 'retirementDateSpecified')
          const retirementAge = {
            inMonths: _.clamp(
              defaultPerson.ages.retirementAge.inMonths,
              fGet(planParamsNorm.ages[personType]).currentAge.inMonths + 1,
              currPerson.ages.maxAge.inMonths - 1,
            ),
          }
          currPerson.ages = {
            type: 'retirementDateSpecified',
            monthOfBirth: currPerson.ages.monthOfBirth,
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
    // SetPersonMonthOfBirth
    // ---------
    case 'setPersonMonthOfBirth2': {
      const { person, monthOfBirth } = action.value
      return {
        applyToClone: (clone) => {
          getPerson(clone, person).ages.monthOfBirth = monthOfBirth
        },
        render: () =>
          `Set ${yourOrYourPartners(
            person,
          )} month of birth to ${CalendarMonthFns.toStr(monthOfBirth)}`,
        merge: (prev) =>
          prev.type === 'setPersonMonthOfBirth' && prev.value.person === person,
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
    // addLabeledAmountTimed
    // ---------
    case 'addLabeledAmountTimed': {
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
    // setMonthRangeForLabeledAmountTimed
    // ---------
    case 'setMonthRangeForLabeledAmountTimed': {
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
          clone.wealth.portfolioBalance = {
            updatedHere: true,
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
        render: () => `Set time preference to ${formatPercentage(1)(-value)}`,
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
    // SetSPAWAndSWRAllocation
    // ---------
    case 'setSPAWAndSWRAllocation': {
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
    // SetExpectedReturns2
    // ---------
    case 'setExpectedReturns2': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.advanced.expectedReturnsForPlanning = value
        },
        render: () => {
          const labelInfo = expectedReturnTypeLabelInfo(value)
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
        merge: (prev) =>
          prev.type === 'setExpectedReturns' &&
          prev.value.type === 'manual' &&
          value.type === 'manual',
      }
    }
    // ---------
    // setHistoricalStockReturnsAdjustmentVolatilityScale
    // ---------
    case 'setHistoricalStockReturnsAdjustmentVolatilityScale': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.advanced.historicalMonthlyLogReturnsAdjustment.standardDeviation.stocks.scale =
            value
        },

        render: () => `Set stocks volatility scaling to ${value}`,
        merge: () => true,
      }
    }

    // ---------
    // setHistoricalBondReturnsAdjustmentEnableVolatility
    // ---------
    case 'setHistoricalBondReturnsAdjustmentEnableVolatility': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.advanced.historicalMonthlyLogReturnsAdjustment.standardDeviation.bonds.enableVolatility =
            value
        },

        render: () => `Bond volatility ${value ? 'enabled' : 'disabled'}`,
        merge: false,
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
        applyToClone: (clone, __, defaultPlanParams: PlanParams) => {
          clone.advanced.sampling = defaultPlanParams.advanced.sampling
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
          clone.advanced.sampling.type = value
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
          clone.advanced.sampling.forMonteCarlo.blockSize = value
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
          clone.advanced.sampling.forMonteCarlo.staggerRunStarts = value
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
    // setHistoricalMonthlyLogReturnsAdjustmentOverrideToFixedForTesting
    // ---------
    case 'setHistoricalMonthlyLogReturnsAdjustmentOverrideToFixedForTesting': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.advanced.historicalMonthlyLogReturnsAdjustment.overrideToFixedForTesting =
            value
        },
        render: () =>
          `Set historical monthly log returns adjustment override to fixed for testing to ${value}`,
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
