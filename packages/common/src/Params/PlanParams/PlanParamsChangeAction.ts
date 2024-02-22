import {
  JSONGuard,
  boolean,
  constant,
  nullable,
  number,
  object,
  string,
  union,
} from 'json-guard'
import { PlanParams21 as V21 } from './Old/PlanParams21'
import { PlanParams22 as V22 } from './Old/PlanParams22'
import { PlanParams23 as V23 } from './Old/PlanParams23'
import { PlanParams26 as V26 } from './Old/PlanParams26'
import {
  PlanParamsChangeActionDeprecated,
  planParamsChangeActionGuardDeprecated,
} from './PlanParamsChangeActionDeprecated'

type _PersonType = 'person1' | 'person2'

// ------------------------------------------
//                   TYPES
// ------------------------------------------

// WARNING: There should not be any dependency on PlanParams but only to
// PlanParamsV since these types are frozen.

export type PlanParamsChangeActionCurrent =
  | { type: 'start'; value: null }
  | { type: 'startCopiedFromBeforeHistory'; value: null }
  | { type: 'startCutByClient'; value: null }
  | { type: 'startFromURL'; value: null }
  | {
      type: 'setDialogPosition'
      value: V21.PlanParams['dialogPosition']
    }
  | { type: 'noOpToMarkMigration'; value: null }
  | { type: 'addPartner'; value: null }
  | { type: 'deletePartner'; value: null }
  | { type: 'setPersonRetired'; value: _PersonType }
  | { type: 'setPersonNotRetired'; value: _PersonType }
  | {
      type: 'setPersonMonthOfBirth'
      value: { person: _PersonType; monthOfBirth: V21.CalendarMonth }
    }
  | {
      type: 'setPersonRetirementAge'
      value: { person: _PersonType; retirementAge: V21.InMonths }
    }
  | {
      type: 'setPersonMaxAge'
      value: { person: _PersonType; maxAge: V21.InMonths }
    }
  | { type: 'setWithdrawalStart'; value: _PersonType }

  // --------------- WEALTH
  | {
      type: 'setCurrentPortfolioBalance'
      value: number
    }
  | {
      type: 'addValueForMonthRange'
      value: {
        location: ValueForMonthRangeLocation
        entryId: string
        sortIndex: number
        monthRange: V21.MonthRange
      }
    }
  | {
      type: 'deleteLabeledAmount'
      value: {
        location: LabeledAmountLocation
        entryId: string
      }
    }
  | {
      type: 'addLabeledAmount'
      value: {
        location: LabeledAmountLocation
        entryId: string
        sortIndex: number
      }
    }
  | {
      type: 'setLabelForLabeledAmount'
      value: {
        location: LabeledAmountLocation
        entryId: string
        label: string | null
      }
    }
  | {
      type: 'setAmountForLabeledAmount'
      value: {
        location: LabeledAmountLocation
        entryId: string
        amount: number
      }
    }
  | {
      type: 'setNominalForLabeledAmount'
      value: {
        location: LabeledAmountLocation
        entryId: string
        nominal: boolean
      }
    }
  | {
      type: 'setMonthRangeForValueForMonthRange'
      value: {
        location: ValueForMonthRangeLocation
        entryId: string
        monthRange: V21.MonthRange
      }
    }
  | { type: 'setSpendingCeiling'; value: number | null }
  | { type: 'setLegacyTotal'; value: number }

  // --------- RISK
  | { type: 'setTPAWRiskTolerance'; value: number }
  | {
      type: 'setTPAWRiskDeltaAtMaxAge'
      value: number
    }
  | {
      type: 'setTPAWRiskToleranceForLegacyAsDeltaFromAt20'
      value: number
    }
  | { type: 'setTPAWTimePreference'; value: number }
  | {
      type: 'setTPAWAdditionalSpendingTilt'
      value: number
    }
  | {
      type: 'setSWRWithdrawalAsPercentPerYear'
      value: number
    }
  | {
      type: 'setSWRWithdrawalAsAmountPerMonth'
      value: number
    }
  | {
      type: 'setSPAWAndSWRAllocation'
      value: V21.GlidePath
    }
  | {
      type: 'setSPAWAnnualSpendingTilt'
      value: number
    }
  | { type: 'setTPAWAndSPAWLMP'; value: number }
  | { type: 'setSpendingFloor'; value: number | null }

  // ----------ADVANCED
  | {
      type: 'setStrategy'
      value: V21.PlanParams['advanced']['strategy']
    }
  | { type: 'setSamplingToDefault'; value: null }
  | { type: 'setSampling'; value: 'historical' | 'monteCarlo' }
  | {
      type: 'setMonteCarloSamplingBlockSize'
      value: number
    }
  | {
      type: 'setMonteCarloStaggerRunStarts'
      value: boolean
    }
  | {
      type: 'setExpectedReturns2'
      value: V26.PlanParams['advanced']['expectedAnnualReturnForPlanning']
    }
  | {
      type: 'setAnnualInflation'
      value: V21.PlanParams['advanced']['annualInflation']
    }
  | {
      type: 'setHistoricalStockReturnsAdjustmentVolatilityScale'
      value: number
    }
  | {
      type: 'setHistoricalBondReturnsAdjustmentEnableVolatility'
      value: boolean
    }

  // -------------- DEV
  | {
      type: 'setHistoricalMonthlyLogReturnsAdjustmentOverrideToFixedForTesting'
      value: boolean
    }

export type PlanParamsChangeAction =
  | PlanParamsChangeActionCurrent
  | PlanParamsChangeActionDeprecated
// ------------------------------------------
//                   GUARDS
// ------------------------------------------
// These guards are not complete. Mostly a sanity check on the shape.
const v21CG = V21.componentGuards
const v22CG = V22.componentGuards
const v23CG = V23.componentGuards
const v26CG = V26.componentGuards
const valueForMonthRangeLocation: JSONGuard<ValueForMonthRangeLocation> = union(
  constant('futureSavings'),
  constant('incomeDuringRetirement'),
  constant('extraSpendingEssential'),
  constant('extraSpendingDiscretionary'),
)
const labeledAmountLocation: JSONGuard<LabeledAmountLocation> = union(
  valueForMonthRangeLocation,
  constant('legacyExternalSources'),
)

const _guard = <T extends string, V>(
  type: T,
  valueGuard: JSONGuard<V>,
): JSONGuard<{ type: T; value: V }> =>
  object({ type: constant(type), value: valueGuard })

export const planParamsChangeActionGuardCurrent: JSONGuard<PlanParamsChangeActionCurrent> =
  union(
    _guard('start', constant(null)),
    _guard('startCopiedFromBeforeHistory', constant(null)),
    _guard('startCutByClient', constant(null)),
    _guard('startFromURL', constant(null)),
    _guard('setDialogPosition', v21CG.dialogPosition(null)),
    _guard('noOpToMarkMigration', constant(null)),
    _guard('addPartner', constant(null)),
    _guard('deletePartner', constant(null)),
    _guard('setPersonRetired', v21CG.personType),
    _guard('setPersonNotRetired', v21CG.personType),
    _guard(
      'setPersonMonthOfBirth',
      object({ person: v21CG.personType, monthOfBirth: v21CG.calendarMonth }),
    ),
    _guard(
      'setPersonRetirementAge',
      object({ person: v21CG.personType, retirementAge: v21CG.inMonths }),
    ),
    _guard(
      'setPersonMaxAge',
      object({ person: v21CG.personType, maxAge: v21CG.inMonths }),
    ),
    _guard('setWithdrawalStart', v21CG.personType),

    // ----------- WEALTH
    _guard('setCurrentPortfolioBalance', number),
    _guard(
      'addValueForMonthRange',
      object({
        location: valueForMonthRangeLocation,
        entryId: string,
        sortIndex: number,
        monthRange: v21CG.monthRange(null),
      }),
    ),
    _guard(
      'deleteLabeledAmount',
      object({ location: labeledAmountLocation, entryId: string }),
    ),
    _guard(
      'addLabeledAmount',
      object({
        location: labeledAmountLocation,
        entryId: string,
        sortIndex: number,
      }),
    ),
    _guard(
      'setLabelForLabeledAmount',
      object({
        location: labeledAmountLocation,
        entryId: string,
        label: nullable(string),
      }),
    ),
    _guard(
      'setAmountForLabeledAmount',
      object({
        location: labeledAmountLocation,
        entryId: string,
        amount: number,
      }),
    ),
    _guard(
      'setNominalForLabeledAmount',
      object({
        location: labeledAmountLocation,
        entryId: string,
        nominal: boolean,
      }),
    ),
    _guard(
      'setMonthRangeForValueForMonthRange',
      object({
        location: valueForMonthRangeLocation,
        entryId: string,
        monthRange: v21CG.monthRange(null),
      }),
    ),
    _guard('setSpendingCeiling', nullable(number)),
    _guard('setSpendingFloor', nullable(number)),
    _guard('setLegacyTotal', number),

    // ------------ RISK
    _guard('setTPAWRiskTolerance', number),
    _guard('setTPAWRiskDeltaAtMaxAge', number),
    _guard('setTPAWRiskToleranceForLegacyAsDeltaFromAt20', number),
    _guard('setTPAWTimePreference', number),
    _guard('setTPAWAdditionalSpendingTilt', number),
    _guard('setSWRWithdrawalAsPercentPerYear', number),
    _guard('setSWRWithdrawalAsAmountPerMonth', number),
    _guard('setSPAWAndSWRAllocation', v21CG.glidePath(null)),
    _guard('setSPAWAnnualSpendingTilt', number),
    _guard('setTPAWAndSPAWLMP', number),

    // ------------ ADVANCED
    _guard('setStrategy', v21CG.strategy),
    _guard('setSamplingToDefault', constant(null)),
    _guard('setSampling', v22CG.samplingType),
    _guard('setMonteCarloSamplingBlockSize', number),
    _guard('setMonteCarloStaggerRunStarts', boolean),
    _guard('setExpectedReturns2', v26CG.expectedAnnualReturnForPlanning),
    _guard('setAnnualInflation', v21CG.annualInflation),
    _guard('setHistoricalStockReturnsAdjustmentVolatilityScale', number),
    _guard('setHistoricalBondReturnsAdjustmentEnableVolatility', boolean),

    // -------------- DEV
    _guard(
      'setHistoricalMonthlyLogReturnsAdjustmentOverrideToFixedForTesting',
      boolean,
    ),
  )

export const planParamsChangeActionGuard: JSONGuard<PlanParamsChangeAction> =
  union(
    planParamsChangeActionGuardCurrent,
    planParamsChangeActionGuardDeprecated,
  )

// ------------------------------------------
//                   HELPERS
// ------------------------------------------
type ValueForMonthRangeLocation =
  | 'futureSavings'
  | 'incomeDuringRetirement'
  | 'extraSpendingEssential'
  | 'extraSpendingDiscretionary'

type LabeledAmountLocation =
  | ValueForMonthRangeLocation
  | 'legacyExternalSources'
