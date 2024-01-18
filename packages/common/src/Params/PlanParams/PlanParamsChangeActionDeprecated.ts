import { JSONGuard, boolean, number, union, object, constant } from 'json-guard'
import { PlanParams21 as V21 } from './Old/PlanParams21'
import { PlanParams22 as V22 } from './Old/PlanParams22'

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

const _guard = <T extends string, V>(
  type: T,
  valueGuard: JSONGuard<V>,
): JSONGuard<{ type: T; value: V }> =>
  object({ type: constant(type), value: valueGuard })

const v22CG = V22.componentGuards
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
  )
