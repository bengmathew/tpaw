import {
  boolean,
  chain,
  constant,
  gt,
  gte,
  integer,
  JSONGuard,
  lt,
  lte,
  number,
  object,
  success,
  union,
} from 'json-guard'
import { Guards } from '../../../Guards'
import { Params20 as NonPlanParamsPrev } from '../../PlanParams/Old/Params20'
export namespace NonPlanParams21 {
  export type NonPlanParams = {
    v: 21
    timezone: { type: 'auto' } | { type: 'manual'; ianaTimezoneName: string }
    percentileRange: { start: number; end: number }
    numOfSimulationForMonteCarloSampling: number
    dev: {
      alwaysShowAllMonths: boolean
    }
  }
  export const currentVersion: NonPlanParams['v'] = 21

  // ----------- GUARD  ---------//
  export const guard: JSONGuard<NonPlanParams> = object({
    v: constant(21),
    timezone: union(
      object({ type: constant('auto') }),
      object({
        type: constant('manual'),
        ianaTimezoneName: Guards.ianaTimezoneName,
      }),
    ),
    percentileRange: object({
      start: chain(number, integer, gte(1), lt(50)),
      end: chain(number, integer, gt(50), lte(99)),
    }),
    numOfSimulationForMonteCarloSampling: chain(number, integer, gt(0)),
    dev: object({
      alwaysShowAllMonths: boolean,
    }),
  })

  export type SomeNonPlanParams =
    | NonPlanParamsPrev.SomePlanParams
    | NonPlanParams

  export const backwardsCompatibleGuard: JSONGuard<SomeNonPlanParams> = (
    x: unknown,
  ) => {
    const result = union(NonPlanParamsPrev.guard, guard)(x)
    return result.error ? result : success(x as SomeNonPlanParams)
  }

  export const migrate = (x: SomeNonPlanParams): NonPlanParams => {
    if ('v' in x && x.v === 21) return x
    const previous: NonPlanParamsPrev.Params =
      'v' in x && x.v === 20 ? x : NonPlanParamsPrev.migrate(x)

    const result = {
      v: 21,
      timezone: { type: 'auto' },
      numOfSimulationForMonteCarloSampling:
        previous.plan.advanced.monteCarloSampling.numOfSimulations,
      percentileRange: {
        start: previous.nonPlan.percentileRange.start,
        end: previous.nonPlan.percentileRange.end,
      },
      dev: {
        alwaysShowAllMonths: previous.nonPlan.dev.alwaysShowAllMonths,
      },
    }
    return guard(result).force()
  }
}
