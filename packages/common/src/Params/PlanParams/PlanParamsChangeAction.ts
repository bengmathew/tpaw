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
import { PlanParams28 as V28 } from './Old/PlanParams28'
import { PlanParams29 as V29 } from './Old/PlanParams29'
import { PlanParams30 as V30 } from './PlanParams30'
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
  | { type: 'setMarketDataDay'; value: V29.CalendarDay }
  | {
      type: 'setDialogPosition'
      value: V21.PlanParams['dialogPosition']
    }
  | { type: 'noOpToMarkMigration'; value: null }

  // --------------- PEOPLE
  | { type: 'addPartner'; value: null }
  | { type: 'deletePartner'; value: null }
  | {
      type: 'setPersonCurrentAgeInfo'
      value: { personId: V29.PersonId; currentAgeInfo: V29.CurrentAgeInfo }
    }
  | { type: 'setPersonRetired'; value: _PersonType }
  | { type: 'setPersonNotRetired'; value: _PersonType }
  | {
      type: 'setPersonRetirementAge'
      value: { person: _PersonType; retirementAge: V21.InMonths }
    }
  | {
      type: 'setPersonMaxAge'
      value: { person: _PersonType; maxAge: V21.InMonths }
    }
  | { type: 'setWithdrawalStart'; value: _PersonType }

  // --------------- LABELED AMOUNT TIMED/UNTIMED
  | {
      type: 'addLabeledAmountUntimed'
      value: {
        location: V28.LabeledAmountUntimedLocation
        entryId: string
        sortIndex: number
      }
    }
  | {
      type: 'addLabeledAmountTimed2'
      value: {
        location: V29.LabeledAmountTimedLocation
        entryId: string
        sortIndex: number
        amountAndTiming: V29.LabeledAmountTimed['amountAndTiming']
      }
    }
  | {
      type: 'deleteLabeledAmountTimedOrUntimed'
      value: {
        location: V28.LabeledAmountTimedOrUntimedLocation
        entryId: string
      }
    }
  | {
      type: 'setLabelForLabeledAmountTimedOrUntimed'
      value: {
        location: V28.LabeledAmountTimedOrUntimedLocation
        entryId: string
        label: string | null
      }
    }
  | {
      type: 'setAmountForLabeledAmountUntimed'
      value: {
        location: V28.LabeledAmountUntimedLocation
        entryId: string
        amount: number
      }
    }
  | {
      type: 'setBaseAmountForLabeledAmountTimed'
      value: {
        location: V28.LabeledAmountTimedLocation
        entryId: string
        baseAmount: number
      }
    }
  | {
      type: 'setNominalForLabeledAmountTimedOrUntimed'
      value: {
        location: V28.LabeledAmountTimedOrUntimedLocation
        entryId: string
        nominal: boolean
      }
    }
  | {
      type: 'setMonthRangeForLabeledAmountTimed2'
      value: {
        location: V29.LabeledAmountTimedLocation
        entryId: string
        monthRange: V29.MonthRange
      }
    }

  // --------------- WEALTH
  | {
      type: 'setCurrentPortfolioBalance'
      value: number
    }
  | { type: 'setSpendingCeiling'; value: number | null }
  | { type: 'setSpendingFloor'; value: number | null }
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
      type: 'setSPAWAndSWRAllocation2'
      value: V29.GlidePath
    }
  | {
      type: 'setSPAWAnnualSpendingTilt'
      value: number
    }
  | { type: 'setTPAWAndSPAWLMP'; value: number }

  // ----------ADVANCED
  | {
      type: 'setExpectedReturnsForPlanning'
      value: V29.PlanParams['advanced']['returnsStatsForPlanning']['expectedValue']['empiricalAnnualNonLog']
    }
  | {
      type: 'setReturnsStatsForPlanningStockVolatilityScale'
      value: number
    }
  | { type: 'setHistoricalReturnsAdjustmentBondVolatilityScale'; value: number }
  | { type: 'setSamplingToDefault'; value: null }
  | { type: 'setSampling'; value: 'historical' | 'monteCarlo' }
  | {
      type: 'setMonteCarloSamplingBlockSize2'
      value: { inMonths: number }
    }
  | {
      type: 'setMonteCarloStaggerRunStarts'
      value: boolean
    }
  | {
      type: 'setAnnualInflation'
      value: V21.PlanParams['advanced']['annualInflation']
    }
  | {
      type: 'setStrategy'
      value: V21.PlanParams['advanced']['strategy']
    }

  // -------------- DEV
  | {
      type: 'setHistoricalMonthlyLogReturnsAdjustmentOverrideToFixedForTesting2'
      value: V30.PlanParams['advanced']['historicalReturnsAdjustment']['overrideToFixedForTesting']
    }

export type PlanParamsChangeAction =
  | PlanParamsChangeActionCurrent
  | PlanParamsChangeActionDeprecated
;``
// ------------------------------------------
//                   GUARDS
// ------------------------------------------
// These guards are not complete. Mostly a sanity check on the shape.
const v21CG = V21.componentGuards
const v22CG = V22.componentGuards
const v28CG = V28.componentGuards
const v29CG = V29.componentGuards
const v30CG = V30.componentGuards

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
    _guard('setMarketDataDay', v29CG.calendarDay),
    _guard('setDialogPosition', v21CG.dialogPosition(null)),
    _guard('noOpToMarkMigration', constant(null)),
    // --------------- PEOPLE
    _guard('addPartner', constant(null)),
    _guard('deletePartner', constant(null)),
    _guard(
      'setPersonCurrentAgeInfo',
      object({
        personId: v29CG.personId,
        currentAgeInfo: v29CG.currentAgeInfo(null),
      }),
    ),
    _guard('setPersonRetired', v21CG.personType),
    _guard('setPersonNotRetired', v21CG.personType),

    _guard(
      'setPersonRetirementAge',
      object({ person: v21CG.personType, retirementAge: v21CG.inMonths }),
    ),
    _guard(
      'setPersonMaxAge',
      object({ person: v21CG.personType, maxAge: v21CG.inMonths }),
    ),
    _guard('setWithdrawalStart', v21CG.personType),

    // --------------- LABELED AMOUNT TIMED/UNTIMED
    _guard(
      'addLabeledAmountUntimed',
      object({
        location: v28CG.labeledAmountUntimedLocation,
        entryId: string,
        sortIndex: number,
      }),
    ),
    _guard(
      'addLabeledAmountTimed2',
      object({
        location: v29CG.labeledAmountTimedLocation,
        entryId: string,
        sortIndex: number,
        amountAndTiming: v29CG.amountAndTiming(null),
      }),
    ),
    _guard(
      'deleteLabeledAmountTimedOrUntimed',
      object({
        location: v28CG.labeledAmountTimedOrUntimedLocation,
        entryId: string,
      }),
    ),
    _guard(
      'setLabelForLabeledAmountTimedOrUntimed',
      object({
        location: v28CG.labeledAmountTimedOrUntimedLocation,
        entryId: string,
        label: nullable(string),
      }),
    ),
    _guard(
      'setAmountForLabeledAmountUntimed',
      object({
        location: v28CG.labeledAmountUntimedLocation,
        entryId: string,
        amount: number,
      }),
    ),
    _guard(
      'setBaseAmountForLabeledAmountTimed',
      object({
        location: v28CG.labeledAmountTimedLocation,
        entryId: string,
        baseAmount: number,
      }),
    ),
    _guard(
      'setNominalForLabeledAmountTimedOrUntimed',
      object({
        location: v28CG.labeledAmountTimedOrUntimedLocation,
        entryId: string,
        nominal: boolean,
      }),
    ),
    _guard(
      'setMonthRangeForLabeledAmountTimed2',
      object({
        location: v29CG.labeledAmountTimedLocation,
        entryId: string,
        monthRange: v29CG.monthRange(null),
      }),
    ),

    // ----------- WEALTH
    _guard('setCurrentPortfolioBalance', number),
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
    _guard('setSPAWAndSWRAllocation2', v29CG.glidePath(null)),
    _guard('setSPAWAnnualSpendingTilt', number),
    _guard('setTPAWAndSPAWLMP', number),

    // ------------ ADVANCED
    _guard('setExpectedReturnsForPlanning', v29CG.expectedReturnsForPlanning),
    _guard('setReturnsStatsForPlanningStockVolatilityScale', number),
    _guard('setHistoricalReturnsAdjustmentBondVolatilityScale', number),
    _guard('setSamplingToDefault', constant(null)),
    _guard('setSampling', v22CG.samplingType),
    _guard('setMonteCarloSamplingBlockSize2', object({ inMonths: number })),
    _guard('setMonteCarloStaggerRunStarts', boolean),
    _guard('setAnnualInflation', v21CG.annualInflation),
    _guard('setStrategy', v21CG.strategy),

    // -------------- DEV
    _guard(
      'setHistoricalMonthlyLogReturnsAdjustmentOverrideToFixedForTesting2',
      v30CG.historicalReturnsAdjustment.overrideToFixedForTesting,
    ),
  )

export const planParamsChangeActionGuard: JSONGuard<PlanParamsChangeAction> =
  union(
    planParamsChangeActionGuardCurrent,
    planParamsChangeActionGuardDeprecated,
  )
