import {
  DEFAULT_ANNUAL_SWR_WITHDRAWAL_PERCENT,
  Month,
  PlanParams,
  PlanParamsChangeAction,
  PlanPaths,
  assert,
  assertFalse,
  block,
  fGet,
  noCase,
} from '@tpaw/common'
import _ from 'lodash'
import {
  PlanParamsExtended,
  getIsFutureSavingsAllowed,
} from '../../../TPAWSimulator/ExtentPlanParams'
import { calendarMonthStr } from '../../../Utils/CalendarMonthStr'
import { formatCurrency } from '../../../Utils/FormatCurrency'
import { formatPercentage } from '../../../Utils/FormatPercentage'
import { numMonthsStr } from '../../../Utils/NumMonthsStr'
import { yourOrYourPartners } from '../../../Utils/YourOrYourPartners'
import { optGet } from '../../../Utils/optGet'
import { nextPlanSectionDialogPosition } from '../Plan/PlanInput/Helpers/PlanSectionDialogPosition'
import { planSectionLabel } from '../Plan/PlanInput/Helpers/PlanSectionLabel'
import { expectedReturnTypeLabel } from '../Plan/PlanInput/PlanInputExpectedReturns'
import { inflationTypeLabel } from '../Plan/PlanInput/PlanInputInflation'

type _ActionFns = {
  applyToClone: (
    clone: PlanParams,
    planParamsExt: PlanParamsExtended,
    defaultPlanParams: PlanParams,
  ) => void
  render: (
    prevPlanParams: PlanParams,
    planParams: PlanParams,
  ) => React.ReactNode
  getBaseURL: (planPath: PlanPaths) => URL
  merge: null | ((prev: PlanParamsChangeAction) => boolean)
}

export const processPlanParamsChangeAction = (
  action: PlanParamsChangeAction,
): _ActionFns => {
  switch (action.type) {
    case 'start':
      return {
        applyToClone: () => assertFalse(),
        render: () => assertFalse(),
        getBaseURL: () => assertFalse(),
        merge: null,
      }
    case 'startCopiedFromBeforeHistory':
      return {
        applyToClone: () => assertFalse(),
        render: () => assertFalse(),
        getBaseURL: () => assertFalse(),
        merge: null,
      }
    case 'startCutByClient':
      return {
        applyToClone: () => assertFalse(),
        render: () => assertFalse(),
        getBaseURL: () => assertFalse(),
        merge: null,
      }
    case 'startFromURL':
      return {
        applyToClone: () => assertFalse(),
        render: () => assertFalse(),
        getBaseURL: () => assertFalse(),
        merge: null,
      }
    case 'noOpToMarkMigration':
      return {
        applyToClone: () => {}, // Do nothing.
        render: () => `Accepted migration to new inputs`,
        getBaseURL: (planPaths) => planPaths(),
        merge: null,
      }
    // ---------
    // SetDialogPosition
    // ---------
    case 'setDialogPosition': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.dialogPosition = value
        },
        render: (prevParams) => {
          switch (prevParams.dialogPosition) {
            case 'age':
            case 'current-portfolio-balance':
            case 'future-savings':
            case 'income-during-retirement':
              return `Completed "${planSectionLabel(
                prevParams.dialogPosition,
              )}" section`
            case 'done':
              return assertFalse()
            case 'show-results':
              return 'Show Results'
            case 'show-all-inputs':
              return 'Show All Inputs'
            default:
              noCase(prevParams.dialogPosition)
          }
        },
        getBaseURL: (planPaths) => planPaths(),
        merge: null,
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
        getBaseURL: (planPaths) => planPaths.age(),
        merge: null,
      }
    }

    // ---------
    // DeletePartner
    // ---------
    case 'deletePartner': {
      return {
        applyToClone: (clone, planParamsExt: PlanParamsExtended) => {
          assert(clone.people.withPartner)
          // There might be references in the past, that can be converted to
          // a calendar month, and references in glide paths that are not currently
          // active for the strategy that won't show up in the validation
          // errors.
          _convertPerson2MonthReferencesToCalendarMonth(clone, planParamsExt)
          clone.people = {
            withPartner: false,
            person1: clone.people.person1,
          }
        },
        render: () => `Deleted partner`,
        getBaseURL: (planPaths) => planPaths.age(),
        merge: null,
      }
    }
    // ---------
    // SetPersonRetired
    // ---------
    case 'setPersonRetired': {
      const personType = action.value
      return {
        applyToClone: (clone, planParamsExt: PlanParamsExtended) => {
          const { isPersonRetired } = planParamsExt
          const isFutureSavingsGoingToBeAllowed = getIsFutureSavingsAllowed(
            personType === 'person1' ? true : isPersonRetired('person1'),
            personType === 'person2'
              ? true
              : clone.people.withPartner
              ? isPersonRetired('person2')
              : undefined,
          )
          if (!isFutureSavingsGoingToBeAllowed) {
            clone.wealth.futureSavings = {}
          }
          _convertRetirementReferencesToNow(clone, planParamsExt, personType)
          if (
            clone.dialogPosition === 'future-savings' &&
            !isFutureSavingsGoingToBeAllowed
          ) {
            clone.dialogPosition = nextPlanSectionDialogPosition(
              'future-savings',
              1, // 0 throws the calculations off.
            )
          }
          const person = _getPerson(clone, personType)
          person.ages = {
            type: 'retiredWithNoRetirementDateSpecified',
            monthOfBirth: person.ages.monthOfBirth,
            maxAge: person.ages.maxAge,
          }
        },
        render: () =>
          `Marked ${
            personType === 'person1' ? 'yourself' : 'your partner'
          } as retired`,
        getBaseURL: (planPaths) => planPaths.age(),
        merge: null,
      }
    }

    // ---------
    // SetPersonNotRetired
    // ---------
    case 'setPersonNotRetired': {
      const personType = action.value
      return {
        applyToClone: (
          clone,
          planParamsExt: PlanParamsExtended,
          defaultPlanParams: PlanParams,
        ) => {
          const { getCurrentAgeOfPerson } = planParamsExt
          const currPerson = _getPerson(clone, personType)
          const defaultPerson = defaultPlanParams.people.person1
          assert(defaultPerson.ages.type === 'retirementDateSpecified')
          const retirementAge =
            defaultPerson.ages.retirementAge.inMonths <=
            getCurrentAgeOfPerson(personType).inMonths
              ? {
                  inMonths: Math.floor(
                    (currPerson.ages.maxAge.inMonths +
                      getCurrentAgeOfPerson(personType).inMonths) /
                      2,
                  ),
                }
              : defaultPerson.ages.retirementAge
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
        getBaseURL: (planPaths) => planPaths.age(),
        merge: null,
      }
    }
    // ---------
    // SetPersonMonthOfBirth
    // ---------
    case 'setPersonMonthOfBirth': {
      const { person, monthOfBirth } = action.value
      return {
        applyToClone: (clone) => {
          _getPerson(clone, person).ages.monthOfBirth = monthOfBirth
        },
        render: () =>
          `Set ${yourOrYourPartners(
            person,
          )} month of birth to ${calendarMonthStr(monthOfBirth)}`,
        getBaseURL: (planPaths) => planPaths.age(),
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
          const person = _getPerson(clone, personType)
          assert(person.ages.type === 'retirementDateSpecified')
          person.ages.retirementAge = retirementAge
        },
        render: () =>
          `Set ${yourOrYourPartners(
            personType,
          )} retirement age to ${numMonthsStr(retirementAge.inMonths)}`,
        getBaseURL: (planPaths) => planPaths.age(),
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
          const person = _getPerson(clone, personType)
          person.ages.maxAge = maxAge
        },
        render: () =>
          `Set ${yourOrYourPartners(personType)} max age to ${numMonthsStr(
            maxAge.inMonths,
          )}`,
        getBaseURL: (planPaths) => planPaths.age(),
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
        getBaseURL: (planPaths) => planPaths.age(),
        merge: null,
      }
    }

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
        getBaseURL: (planPaths) => planPaths['current-portfolio-balance'](),
        merge: (prev) => prev.type === 'setCurrentPortfolioBalance',
      }
    }

    // ---------
    // AddValueForMonthRange
    // ---------
    case 'addValueForMonthRange': {
      const { location, entryId, monthRange, sortIndex } = action.value
      return {
        applyToClone: (clone) => {
          assert(
            !optGet(
              getValueForMonthRangeEntriesByLocation(clone, location),
              entryId,
            ),
          )
          getValueForMonthRangeEntriesByLocation(clone, location)[entryId] = {
            id: entryId,
            sortIndex,
            label: null,
            value: 0,
            nominal: false,
            monthRange,
          }
        },
        render: () => {
          return `Added ${_getValueForMonthRangeLocationStr(location)} entry`
        },
        getBaseURL: _getlabeledAmountBaseURLByLocation(location, null),
        merge: null,
      }
    }

    // ---------
    // AddLabeledAmount
    // ---------
    case 'addLabeledAmount': {
      const { location, entryId, sortIndex } = action.value
      return {
        applyToClone: (clone) => {
          assert(
            !optGet(
              getLabeledAmountEntriesByLocation(clone, location),
              entryId,
            ),
          )
          getLabeledAmountEntriesByLocation(clone, location)[entryId] = {
            id: entryId,
            label: null,
            value: 0,
            nominal: false,
            sortIndex,
          }
        },
        render: () => {
          return `Added ${_getLabeledAmountLocationStr(location)} entry `
        },
        getBaseURL: _getlabeledAmountBaseURLByLocation(location, null),
        merge: null,
      }
    }
    // ---------
    // DeleteLabeledAmount
    // ---------
    case 'deleteLabeledAmount': {
      const { location, entryId } = action.value
      return {
        applyToClone: (clone) => {
          const entries = getLabeledAmountEntriesByLocation(clone, location)
          assert(optGet(entries, entryId))
          delete entries[entryId]
        },
        render: (prevParams) => {
          return `Deleted ${_getLabeledAmountLocationStr(
            location,
          )} entry "${_truncateLabel(
            fGet(
              optGet(
                getLabeledAmountEntriesByLocation(prevParams, location),
                entryId,
              ),
            ).label,
          )}" `
        },
        getBaseURL: _getlabeledAmountBaseURLByLocation(location, entryId),
        merge: null,
      }
    }
    // ---------
    // SetLabelForLabeledAmount
    // ---------
    case 'setLabelForLabeledAmount': {
      const { location, entryId, label } = action.value
      return {
        applyToClone: (clone) => {
          fGet(
            optGet(getLabeledAmountEntriesByLocation(clone, location), entryId),
          ).label = label
        },
        render: () => {
          return `Set ${_getLabeledAmountLocationStr(
            location,
          )} entry label to "${_truncateLabel(label)}'" `
        },
        getBaseURL: _getlabeledAmountBaseURLByLocation(location, entryId),
        merge: (prev) =>
          prev.type === 'setLabelForLabeledAmount' &&
          prev.value.entryId === entryId,
      }
    }

    // ---------
    // SetAmountForLabeledAmount
    // ---------
    case 'setAmountForLabeledAmount': {
      const { location, entryId, amount } = action.value
      return {
        applyToClone: (clone) => {
          fGet(
            optGet(getLabeledAmountEntriesByLocation(clone, location), entryId),
          ).value = amount
        },
        render: (_: PlanParams, planParams: PlanParams) => {
          return `Set ${_getLabeledAmountLocationStr(
            location,
          )} entry "${_truncateLabel(
            fGet(
              optGet(
                getLabeledAmountEntriesByLocation(planParams, location),
                entryId,
              ),
            ).label,
          )}" amount to ${formatCurrency(amount)} per month`
        },

        getBaseURL: _getlabeledAmountBaseURLByLocation(location, entryId),
        merge: (prev) =>
          prev.type === 'setAmountForLabeledAmount' &&
          prev.value.entryId === entryId,
      }
    }

    // ---------
    // SetNominalForLabeledAmount
    // ---------
    case 'setNominalForLabeledAmount': {
      const { location, entryId, nominal } = action.value
      return {
        applyToClone: (clone) => {
          fGet(
            optGet(getLabeledAmountEntriesByLocation(clone, location), entryId),
          ).nominal = nominal
        },
        render: (_: PlanParams, planParams: PlanParams) => {
          return `Set ${_getLabeledAmountLocationStr(
            location,
          )} entry "${_truncateLabel(
            fGet(
              optGet(
                getLabeledAmountEntriesByLocation(planParams, location),
                entryId,
              ),
            ).label,
          )}" amount as ${nominal ? 'nominal' : 'rea'} dollars`
        },
        getBaseURL: _getlabeledAmountBaseURLByLocation(location, entryId),
        merge: null,
      }
    }
    // ---------
    // SetMonthRangeForValueForMonthRange
    // ---------
    case 'setMonthRangeForValueForMonthRange': {
      const { location, entryId, monthRange } = action.value
      return {
        applyToClone: (clone) => {
          fGet(
            optGet(
              getValueForMonthRangeEntriesByLocation(clone, location),
              entryId,
            ),
          ).monthRange = monthRange
        },
        render: (_: PlanParams, planParams: PlanParams) => {
          return `Set ${_getLabeledAmountLocationStr(
            location,
          )} entry "${_truncateLabel(
            fGet(
              optGet(
                getLabeledAmountEntriesByLocation(planParams, location),
                entryId,
              ),
            ).label,
          )}" month range`
        },
        getBaseURL: _getlabeledAmountBaseURLByLocation(location, entryId),
        merge: (prev) =>
          prev.type === 'setAmountForLabeledAmount' &&
          prev.value.entryId === entryId,
      }
    }

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
        getBaseURL: (planPaths) => planPaths.risk(),
        merge: (prev) => prev.type === 'setTPAWRiskTolerance',
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
          `Set additional spending tilt to ${formatPercentage(1)(value)}`,
        getBaseURL: (planPaths) => planPaths.risk(),
        merge: (prev) => prev.type === 'setTPAWAdditionalSpendingTilt',
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
        getBaseURL: (planPaths) => planPaths.risk(),
        merge: (prev) => prev.type === 'setTPAWRiskDeltaAtMaxAge',
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
        getBaseURL: (planPaths) => planPaths.risk(),
        merge: (prev) =>
          prev.type === 'setTPAWRiskToleranceForLegacyAsDeltaFromAt20',
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
        render: () => `Set spending tilt to ${formatPercentage(1)(-value)}`,
        getBaseURL: (planPaths) => planPaths.risk(),
        merge: (prev) => prev.type === 'setTPAWTimePreference',
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
        getBaseURL: (planPaths) => planPaths.risk(),
        merge: (prev) => prev.type === 'setSWRWithdrawalAsPercentPerYear',
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
        getBaseURL: (planPaths) => planPaths.risk(),
        merge: (prev) => prev.type === 'setSWRWithdrawalAsAmountPerMonth',
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
        getBaseURL: (planPaths) => planPaths.risk(),
        merge: (prev) => prev.type === 'setSPAWAndSWRAllocation',
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
        getBaseURL: (planPaths) => planPaths.risk(),
        merge: (prev) => prev.type === 'setSPAWAnnualSpendingTilt',
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
        getBaseURL: (planPaths) => planPaths.risk(),
        merge: (prev) => prev.type === 'setTPAWAndSPAWLMP',
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
        getBaseURL: (planPaths) => planPaths['spending-ceiling-and-floor'](),
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
        getBaseURL: (planPaths) => planPaths['spending-ceiling-and-floor'](),
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
        getBaseURL: (planPaths) => planPaths['legacy'](),
        merge: (prev) => prev.type === 'setLegacyTotal',
      }
    }

    // ---------
    // SetExpectedReturns
    // ---------
    case 'setExpectedReturns': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.advanced.annualReturns.expected = value
        },
        render: () => {
          return `Set expected returns to ${_.lowerFirst(
            expectedReturnTypeLabel(value),
          )}${
            value.type === 'manual'
              ? ` (stocks: ${formatPercentage(1)(
                  value.stocks,
                )}, bonds: ${formatPercentage(1)(value.bonds)})`
              : ''
          }`
        },
        getBaseURL: (planPaths) => planPaths['expected-returns'](),
        merge: (prev) =>
          prev.type === 'setExpectedReturns' &&
          prev.value.type === 'manual' &&
          value.type === 'manual',
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
        getBaseURL: (planPaths) => planPaths['inflation'](),
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
        applyToClone: (
          clone,
          planParamsExt: PlanParamsExtended,
          defaultPlanParams: PlanParams,
        ) => {
          clone.advanced.sampling = defaultPlanParams.advanced.sampling
          clone.advanced.monteCarloSampling.blockSize =
            defaultPlanParams.advanced.monteCarloSampling.blockSize
        },
        render: (_: PlanParams, planParams: PlanParams) =>
          `Set simulation to Monte Carlo sequence`,
        getBaseURL: (planPaths) => planPaths['simulation'](),
        merge: null,
      }
    }

    // ---------
    // SetSampling
    // ---------
    case 'setSampling': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.advanced.sampling = value
        },
        render: () =>
          `Set simulation to ${
            value === 'historical' ? 'historical' : ' Monte Carlo'
          } sequence`,
        getBaseURL: (planPaths) => planPaths['simulation'](),
        merge: null,
      }
    }

    // ---------
    // SetMonteCarloSamplingBlockSize
    // ---------
    case 'setMonteCarloSamplingBlockSize': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.advanced.monteCarloSampling.blockSize = value
        },
        render: () =>
          `Set block size for Monte Carlo simulation to ${numMonthsStr(
            value,
          )}`,
        getBaseURL: (planPaths) => planPaths['simulation'](),
        merge: (prev) => prev.type === 'setMonteCarloSamplingBlockSize',
      }
    }

    // ---------
    // SetStrategy
    // ---------
    case 'setStrategy': {
      const { value } = action
      return {
        applyToClone: (clone, planParamsExt: PlanParamsExtended) => {
          clone.advanced.strategy = value
          if (value === 'SWR' && clone.risk.swr.withdrawal.type === 'default') {
            clone.risk.swr.withdrawal = {
              type: 'asPercentPerYear',
              percentPerYear: DEFAULT_ANNUAL_SWR_WITHDRAWAL_PERCENT(
                planParamsExt.numRetirementMonths,
              ),
            }
          }
        },
        render: () => `Set strategy to ${value}`,
        getBaseURL: (planPaths) => planPaths['strategy'](),
        merge: null,
      }
    }

    // ---------
    // SwitchHistoricalReturns
    // ---------
    case 'switchHistoricalReturns': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.advanced.annualReturns.historical = value
        },
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
        getBaseURL: (planPaths) => planPaths['dev-historical-returns'](),
        merge: null,
      }
    }

    // ---------
    // SetHistoricalReturnsAdjustForBlockSampling
    // ---------
    case 'setHistoricalReturnsAdjustForBlockSampling': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          assert(clone.advanced.annualReturns.historical.type === 'adjusted')
          clone.advanced.annualReturns.historical.correctForBlockSampling =
            value
        },
        render: () =>
          `${
            value ? 'Set' : 'Unset'
          } correct historical returns for block sampling.`,
        getBaseURL: (planPaths) => planPaths['dev-historical-returns'](),
        merge: null,
      }
    }

    // ---------
    // SetHistoricalReturnsFixedStocks
    // ---------
    case 'setHistoricalReturnsFixedStocks': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          assert(clone.advanced.annualReturns.historical.type === 'fixed')
          clone.advanced.annualReturns.historical.stocks = value / 100
        },
        render: () =>
          `Set fixed historical returns stocks to ${formatPercentage(1)(
            value,
          )}`,
        getBaseURL: (planPaths) => planPaths['dev-historical-returns'](),
        merge: (prev) => prev.type === 'setHistoricalReturnsFixedStocks',
      }
    }

    case 'setHistoricalReturnsFixedBonds': {
      // --------- // SetHistoricalReturnsFixedBonds // ---------
      const { value } = action
      return {
        applyToClone: (clone) => {
          assert(clone.advanced.annualReturns.historical.type === 'fixed')
          clone.advanced.annualReturns.historical.bonds = value / 100
        },
        render: () =>
          `Set fixed historical returns bonds to ${formatPercentage(1)(value)}`,
        getBaseURL: (planPaths) => planPaths['dev-historical-returns'](),
        merge: (prev) => prev.type === 'setHistoricalReturnsFixedBonds',
      }
    }
    default:
      noCase(action)
  }
}

type ValueForMonthRangeLocation =
  | 'futureSavings'
  | 'incomeDuringRetirement'
  | 'extraSpendingEssential'
  | 'extraSpendingDiscretionary'

type LabeledAmountLocation =
  | ValueForMonthRangeLocation
  | 'legacyExternalSources'
export const getValueForMonthRangeEntriesByLocation = (
  planParams: PlanParams,
  location: ValueForMonthRangeLocation,
) => {
  switch (location) {
    case 'extraSpendingEssential':
      return planParams.adjustmentsToSpending.extraSpending.essential
    case 'extraSpendingDiscretionary':
      return planParams.adjustmentsToSpending.extraSpending.discretionary
    case 'futureSavings':
    case 'incomeDuringRetirement':
      return planParams.wealth[location]
    default:
      noCase(location)
  }
}
const _getValueForMonthRangeLocationStr = (
  location: ValueForMonthRangeLocation,
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

export const getLabeledAmountEntriesByLocation = (
  planParams: PlanParams,
  location: LabeledAmountLocation,
) => {
  switch (location) {
    case 'legacyExternalSources':
      return planParams.adjustmentsToSpending.tpawAndSPAW.legacy.external
    default:
      return getValueForMonthRangeEntriesByLocation(planParams, location)
  }
}

const _getLabeledAmountLocationStr = (location: LabeledAmountLocation) => {
  switch (location) {
    case 'legacyExternalSources':
      return 'legacy non-portfolio source'
    default:
      return _getValueForMonthRangeLocationStr(location)
  }
}

const _getlabeledAmountBaseURLByLocation =
  (location: LabeledAmountLocation, entryId: string | null) =>
  (planPaths: PlanPaths) => {
    const url = block(() => {
      switch (location) {
        case 'futureSavings':
          return planPaths['future-savings']()
        case 'incomeDuringRetirement':
          return planPaths['income-during-retirement']()
        case 'extraSpendingEssential':
        case 'extraSpendingDiscretionary':
          return planPaths['extra-spending']()
        case 'legacyExternalSources':
          return planPaths['legacy']()
        default:
          noCase(location)
      }
    })
    if (entryId) {
      url.searchParams.set('entryId', entryId)
    }
    return url
  }

const _truncateLabel = (label: string | null) =>
  label ? _.truncate(label, { length: 30 }) : '<no label>'

// Uses MasterListOfMonths
const _modifyMonthsInPlanParams = (
  planParams: PlanParams,
  fn: (x: Month) => Month,
) => {
  ;[
    ..._.values(planParams.wealth.futureSavings),
    ..._.values(planParams.wealth.incomeDuringRetirement),
    ..._.values(planParams.adjustmentsToSpending.extraSpending.essential),
    ..._.values(planParams.adjustmentsToSpending.extraSpending.discretionary),
  ].forEach((x) => {
    x.monthRange = block(() => {
      switch (x.monthRange.type) {
        case 'startAndEnd':
          return {
            type: 'startAndEnd',
            start: fn(x.monthRange.start),
            end: fn(x.monthRange.end),
          }
        case 'endAndNumMonths':
          return {
            type: 'endAndNumMonths',
            end: fn(x.monthRange.end),
            numMonths: x.monthRange.numMonths,
          }
        case 'startAndNumMonths':
          return {
            type: 'startAndNumMonths',
            start: fn(x.monthRange.start),
            numMonths: x.monthRange.numMonths,
          }
        default:
          noCase(x.monthRange)
      }
    })
  })
  ;[..._.values(planParams.risk.spawAndSWR.allocation.intermediate)].forEach(
    (x) => {
      x.month = fn(x.month)
    },
  )
}

const _convertPerson2MonthReferencesToCalendarMonth = (
  clone: PlanParams,
  planParamsExt: PlanParamsExtended,
) => {
  const { months, asMFN, monthsFromNowToCalendarMonth } = planParamsExt
  _modifyMonthsInPlanParams(clone, (month) =>
    month.type !== 'calendarMonth' &&
    month.type !== 'calendarMonthAsNow' &&
    month.person === 'person2'
      ? months.calendarMonth(monthsFromNowToCalendarMonth(asMFN(month)))
      : month,
  )
}

const _convertRetirementReferencesToNow = (
  clone: PlanParams,
  planParamsExt: PlanParamsExtended,
  personType: 'person1' | 'person2',
) => {
  const { months } = planParamsExt
  _modifyMonthsInPlanParams(clone, (month) =>
    month.type === 'namedAge' &&
    month.person === personType &&
    (month.age === 'retirement' || month.age === 'lastWorkingMonth')
      ? months.now
      : month,
  )
}

const _getPerson = (params: PlanParams, personType: 'person1' | 'person2') => {
  if (personType === 'person1') return params.people.person1
  assert(params.people.withPartner)
  return params.people.person2
}

export const doesURLMatch = (baseURL: URL, currURL: URL) => {
  const path = currURL.pathname === baseURL.pathname
  const searchParams = [...baseURL.searchParams.keys()].every(
    (key) => currURL.searchParams.get(key) === baseURL.searchParams.get(key),
  )
  return path && searchParams
    ? ({ full: true, path: true, searchParams: true } as const)
    : ({ full: false, path, searchParams } as const)
}