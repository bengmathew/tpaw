import { JSONGuard, boolean, number, union, object, constant } from 'json-guard'
import { PlanParams21 as V21 } from './Old/PlanParams21'

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

const _guard = <T extends string, V>(
  type: T,
  valueGuard: JSONGuard<V>,
): JSONGuard<{ type: T; value: V }> =>
  object({ type: constant(type), value: valueGuard })

export const planParamsChangeActionGuardDeprecated: JSONGuard<PlanParamsChangeActionDeprecated> =
  union(
    _guard(
      'switchHistoricalReturns',
      V21.componentGuards.historicalAnnualReturns,
    ),
    _guard('setHistoricalReturnsAdjustForBlockSampling', boolean),
    _guard('setHistoricalReturnsFixedStocks', number),
    _guard('setHistoricalReturnsFixedBonds', number),
  )
