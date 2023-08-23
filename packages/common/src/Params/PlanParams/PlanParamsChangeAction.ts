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
import {
  CalendarMonth,
  GlidePath,
  InMonths,
  MonthRange,
  PlanParams,
  planParamsComponentGuards,
} from './PlanParams'

type _PersonType = 'person1' | 'person2'

// ------------------------------------------
//                   TYPES
// ------------------------------------------

export type PlanParamsChangeAction =
  | { type: 'start'; value: null }
  | { type: 'startCopiedFromBeforeHistory'; value: null }
  | { type: 'startCutByClient'; value: null }
  | { type: 'startFromURL'; value: null }
  | { type: 'setDialogPosition'; value: PlanParams['dialogPosition'] }
  | { type: 'noOpToMarkMigration'; value: null }
  | { type: 'addPartner'; value: null }
  | { type: 'deletePartner'; value: null }
  | { type: 'setPersonRetired'; value: _PersonType }
  | { type: 'setPersonNotRetired'; value: _PersonType }
  | {
      type: 'setPersonMonthOfBirth'
      value: { person: _PersonType; monthOfBirth: CalendarMonth }
    }
  | {
      type: 'setPersonRetirementAge'
      value: { person: _PersonType; retirementAge: InMonths }
    }
  | {
      type: 'setPersonMaxAge'
      value: { person: _PersonType; maxAge: InMonths }
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
        monthRange: MonthRange
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
        monthRange: MonthRange
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
      value: GlidePath
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
      value: PlanParams['advanced']['strategy']
    }
  | { type: 'setSamplingToDefault'; value: null }
  | { type: 'setSampling'; value: 'historical' | 'monteCarlo' }
  | {
      type: 'setMonteCarloSamplingBlockSize'
      value: number
    }
  | {
      type: 'setExpectedReturns'
      value: PlanParams['advanced']['annualReturns']['expected']
    }
  | {
      type: 'setAnnualInflation'
      value: PlanParams['advanced']['annualInflation']
    }

  // -------------- DEV
  | {
      type: 'switchHistoricalReturns'
      value: PlanParams['advanced']['annualReturns']['historical']
    }
  | {
      type: 'setHistoricalReturnsAdjustForBlockSampling'
      value: boolean
    }
  | {
      type: 'setHistoricalReturnsFixedStocks'
      value: number
    }
  | {
      type: 'setHistoricalReturnsFixedBonds'
      value: number
    }

// ------------------------------------------
//                   GUARDS
// ------------------------------------------
// These guards are not complete. Mostly a sanity check on the shape.
const cg = planParamsComponentGuards
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

export const planParamsChangeActionGuard: JSONGuard<PlanParamsChangeAction> =
  union(
    _guard('start', constant(null)),
    _guard('startCopiedFromBeforeHistory', constant(null)),
    _guard('startCutByClient', constant(null)),
    _guard('startFromURL', constant(null)),
    _guard('setDialogPosition', cg.dialogPosition(null)),
    _guard('noOpToMarkMigration', constant(null)),
    _guard('addPartner', constant(null)),
    _guard('deletePartner', constant(null)),
    _guard('setPersonRetired', cg.personType),
    _guard('setPersonNotRetired', cg.personType),
    _guard(
      'setPersonMonthOfBirth',
      object({ person: cg.personType, monthOfBirth: cg.calendarMonth }),
    ),
    _guard(
      'setPersonRetirementAge',
      object({ person: cg.personType, retirementAge: cg.inMonths }),
    ),
    _guard(
      'setPersonMaxAge',
      object({ person: cg.personType, maxAge: cg.inMonths }),
    ),
    _guard('setWithdrawalStart', cg.personType),

    // ----------- WEALTH
    _guard('setCurrentPortfolioBalance', number),
    _guard(
      'addValueForMonthRange',
      object({
        location: valueForMonthRangeLocation,
        entryId: string,
        sortIndex: number,
        monthRange: cg.monthRange(null),
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
        monthRange: cg.monthRange(null),
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
    _guard('setSPAWAndSWRAllocation', cg.glidePath(null)),
    _guard('setSPAWAnnualSpendingTilt', number),
    _guard('setTPAWAndSPAWLMP', number),

    // ------------ ADVANCED
    _guard('setStrategy', cg.strategy),
    _guard('setSamplingToDefault', constant(null)),
    _guard('setSampling', cg.sampling),
    _guard('setMonteCarloSamplingBlockSize', number),
    _guard('setExpectedReturns', cg.expectedAnnualReturns),
    _guard('setAnnualInflation', cg.annualInflation),

    // -------------- DEV
    _guard('switchHistoricalReturns', cg.historicalAnnualReturns),
    _guard('setHistoricalReturnsAdjustForBlockSampling', boolean),
    _guard('setHistoricalReturnsFixedStocks', number),
    _guard('setHistoricalReturnsFixedBonds', number),
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
