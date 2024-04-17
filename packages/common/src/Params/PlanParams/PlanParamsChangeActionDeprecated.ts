import {
  JSONGuard,
  boolean,
  number,
  union,
  object,
  constant,
  string,
  nullable,
} from 'json-guard'
import { PlanParams21 as V21 } from './Old/PlanParams21'
import { PlanParams22 as V22 } from './Old/PlanParams22'
import { PlanParams23 as V23 } from './Old/PlanParams23'
import { PlanParams26 as V26 } from './Old/PlanParams26'
import { PlanParams28 as V28 } from './Old/PlanParams28'

type Pre28ValueForMonthRangeLocation =
  | 'futureSavings'
  | 'incomeDuringRetirement'
  | 'extraSpendingEssential'
  | 'extraSpendingDiscretionary'

type Pre28LabeledAmountLocation =
  | Pre28ValueForMonthRangeLocation
  | 'legacyExternalSources'

export type PlanParamsChangeActionDeprecated =
  | {
      type: 'switchHistoricalReturns'
      value: V21.PlanParams['advanced']['annualReturns']['historical']
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
  | {
      type: 'setHistoricalReturnsBonds'
      value:
        | 'fixedToExpectedUsedForPlanning'
        | 'adjustExpectedToExpectedUsedForPlanning'
    }
  | {
      type: 'setHistoricalReturnsStocksDev'
      value: V22.PlanParams['advanced']['annualReturns']['historical']['stocks']
    }
  | {
      type: 'setHistoricalReturnsBondsDev'
      value: V22.PlanParams['advanced']['annualReturns']['historical']['bonds']
    }
  | {
      type: 'setExpectedReturns'
      value: V21.PlanParams['advanced']['annualReturns']['expected']
    }
  | {
      type: 'setHistoricalReturnsAdjustExpectedReturnDev'
      value: {
        type: 'stocks' | 'bonds'
        adjustExpectedReturn: V23.PlanParams['advanced']['historicalReturnsAdjustment']['stocks']['adjustExpectedReturn']
      }
    }
  // ----------- Deprecated in v28
  | {
      type: 'addValueForMonthRange'
      value: {
        location: Pre28ValueForMonthRangeLocation
        entryId: string
        sortIndex: number
        monthRange: V21.MonthRange
      }
    }
  | {
      type: 'deleteLabeledAmount'
      value: {
        location: Pre28LabeledAmountLocation
        entryId: string
      }
    }
  | {
      type: 'addLabeledAmount'
      value: {
        // This should not have included ValueForMonthRangeLocation, but it did.
        // However this action was never called for any location other than
        // 'legacyExternalSources'.
        location: Pre28LabeledAmountLocation
        entryId: string
        sortIndex: number
      }
    }
  | {
      type: 'setLabelForLabeledAmount'
      value: {
        location: Pre28LabeledAmountLocation
        entryId: string
        label: string | null
      }
    }
  | {
      type: 'setAmountForLabeledAmount'
      value: {
        location: Pre28LabeledAmountLocation
        entryId: string
        amount: number
      }
    }
  | {
      type: 'setNominalForLabeledAmount'
      value: {
        location: Pre28LabeledAmountLocation
        entryId: string
        nominal: boolean
      }
    }
  | {
      type: 'setMonthRangeForValueForMonthRange'
      value: {
        location: Pre28ValueForMonthRangeLocation
        entryId: string
        monthRange: V21.MonthRange
      }
    }
  | {
      type: 'setPersonMonthOfBirth'
      value: { person: 'person1' | 'person2'; monthOfBirth: V21.CalendarMonth }
    }
  | {
      type: 'setMonteCarloSamplingBlockSize'
      value: number
    }
  // -------  Deprecated in v29
  | {
      type: 'setHistoricalStockReturnsAdjustmentVolatilityScale'
      value: number
    }
  | {
      type: 'setHistoricalBondReturnsAdjustmentEnableVolatility'
      value: boolean
    }
  | {
      type: 'setExpectedReturns2'
      value: V26.PlanParams['advanced']['expectedAnnualReturnForPlanning']
    }
  | {
      type: 'setPersonMonthOfBirth2'
      value: { person: 'person1' | 'person2'; monthOfBirth: V28.CalendarMonth }
    }
  | {
      type: 'setSPAWAndSWRAllocation'
      value: V21.GlidePath
    }
  | {
      type: 'setMonthRangeForLabeledAmountTimed'
      value: {
        location: V28.LabeledAmountTimedLocation
        entryId: string
        monthRange: V28.MonthRange
      }
    }
  | {
      type: 'addLabeledAmountTimed'
      value: {
        location: V28.LabeledAmountTimedLocation
        entryId: string
        sortIndex: number
        amountAndTiming: V28.LabeledAmountTimed['amountAndTiming']
      }
    }

// ------- end Deprecated in v29

const _guard = <T extends string, V>(
  type: T,
  valueGuard: JSONGuard<V>,
): JSONGuard<{ type: T; value: V }> =>
  object({ type: constant(type), value: valueGuard })

const pre28ValueForMonthRangeLocationGuard: JSONGuard<Pre28ValueForMonthRangeLocation> =
  union(
    constant('futureSavings'),
    constant('incomeDuringRetirement'),
    constant('extraSpendingEssential'),
    constant('extraSpendingDiscretionary'),
  )

const pre28LabeledAmountLocationGuard: JSONGuard<Pre28LabeledAmountLocation> =
  union(pre28ValueForMonthRangeLocationGuard, constant('legacyExternalSources'))

const v21CG = V21.componentGuards
const v22CG = V22.componentGuards
const v23CG = V23.componentGuards
const v26CG = V26.componentGuards
const v28CG = V28.componentGuards
export const planParamsChangeActionGuardDeprecated: JSONGuard<PlanParamsChangeActionDeprecated> =
  union(
    _guard(
      'switchHistoricalReturns',
      V21.componentGuards.historicalAnnualReturns,
    ),
    _guard('setHistoricalReturnsAdjustForBlockSampling', boolean),
    _guard('setHistoricalReturnsFixedStocks', number),
    _guard('setHistoricalReturnsFixedBonds', number),
    _guard(
      'setHistoricalReturnsBonds',
      union(
        constant('fixedToExpectedUsedForPlanning'),
        constant('adjustExpectedToExpectedUsedForPlanning'),
      ),
    ),
    _guard('setHistoricalReturnsStocksDev', v22CG.historicalAnnualReturns),
    _guard('setHistoricalReturnsBondsDev', v22CG.historicalAnnualReturns),
    _guard('setExpectedReturns', v21CG.expectedAnnualReturns),
    _guard(
      'setHistoricalReturnsAdjustExpectedReturnDev',
      object({
        type: union(constant('stocks'), constant('bonds')),
        adjustExpectedReturn: v23CG.adjustExpectedReturn,
      }),
    ),
    // -------------- Deprecated in v28
    _guard(
      'addValueForMonthRange',
      object({
        location: pre28ValueForMonthRangeLocationGuard,
        entryId: string,
        sortIndex: number,
        monthRange: v21CG.monthRange(null),
      }),
    ),
    _guard(
      'deleteLabeledAmount',
      object({ location: pre28LabeledAmountLocationGuard, entryId: string }),
    ),
    _guard(
      'addLabeledAmount',
      object({
        location: pre28LabeledAmountLocationGuard,
        entryId: string,
        sortIndex: number,
      }),
    ),
    _guard(
      'setLabelForLabeledAmount',
      object({
        location: pre28LabeledAmountLocationGuard,
        entryId: string,
        label: nullable(string),
      }),
    ),
    _guard(
      'setAmountForLabeledAmount',
      object({
        location: pre28LabeledAmountLocationGuard,
        entryId: string,
        amount: number,
      }),
    ),
    _guard(
      'setNominalForLabeledAmount',
      object({
        location: pre28LabeledAmountLocationGuard,
        entryId: string,
        nominal: boolean,
      }),
    ),
    _guard(
      'setMonthRangeForValueForMonthRange',
      object({
        location: pre28ValueForMonthRangeLocationGuard,
        entryId: string,
        monthRange: v21CG.monthRange(null),
      }),
    ),
    _guard(
      'setPersonMonthOfBirth',
      object({ person: v21CG.personType, monthOfBirth: v21CG.calendarMonth }),
    ),
    _guard('setMonteCarloSamplingBlockSize', number),
    // -------  Deprecated in v29
    _guard('setHistoricalStockReturnsAdjustmentVolatilityScale', number),
    _guard('setHistoricalBondReturnsAdjustmentEnableVolatility', boolean),
    _guard('setExpectedReturns2', v26CG.expectedAnnualReturnForPlanning),
    _guard(
      'setPersonMonthOfBirth2',
      object({ person: v21CG.personType, monthOfBirth: v28CG.monthOfBirth }),
    ),
    _guard('setSPAWAndSWRAllocation', v21CG.glidePath(null)),
    _guard(
      'setMonthRangeForLabeledAmountTimed',
      object({
        location: v28CG.labeledAmountTimedLocation,
        entryId: string,
        monthRange: v28CG.monthRange(null),
      }),
    ),
    _guard(
      'addLabeledAmountTimed',
      object({
        location: v28CG.labeledAmountTimedLocation,
        entryId: string,
        sortIndex: number,
        amountAndTiming: v28CG.amountAndTiming(null),
      }),
    ),

    // ------- end Deprecated in v29
  )
