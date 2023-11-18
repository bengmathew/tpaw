import {
  boolean,
  chain,
  constant,
  failure,
  gt,
  integer,
  JSONGuard,
  number,
  object,
  success,
  union,
} from 'json-guard'
import { Guards } from '../../Guards'
import { assert } from '../../Utils'
import { NonPlanParams21 as NonPlanParamsPrev } from './Old/NonPlanParams21'
export namespace NonPlanParams22 {
  export const currentVersion = 22 as const
  export type NonPlanParams = {
    v: typeof currentVersion
    timezone: { type: 'auto' } | { type: 'manual'; ianaTimezoneName: string }
    numOfSimulationForMonteCarloSampling: number
    dev: {
      showDevFeatures: boolean
      alwaysShowAllMonths: boolean
      overridePlanResultChartYRange: false | { start: number; end: number }
    }
  }

  // ----------- GUARD  ---------//
  export const guard: JSONGuard<NonPlanParams> = object({
    v: constant(currentVersion),
    timezone: union(
      object({ type: constant('auto') }),
      object({
        type: constant('manual'),
        ianaTimezoneName: Guards.ianaTimezoneName,
      }),
    ),
    numOfSimulationForMonteCarloSampling: chain(number, integer, gt(0)),
    dev: object({
      showDevFeatures: boolean,
      alwaysShowAllMonths: boolean,
      overridePlanResultChartYRange: union(
        constant(false),
        chain(object({ start: number, end: number }), (x) =>
          x.start < x.end ? success(x) : failure('start must be less than end'),
        ),
      ),
    }),
  })

  export type SomeNonPlanParams =
    | NonPlanParamsPrev.SomeNonPlanParams
    | NonPlanParams

  export const backwardsCompatibleGuard: JSONGuard<SomeNonPlanParams> = (
    x: unknown,
  ) => {
    const result = union(NonPlanParamsPrev.backwardsCompatibleGuard, guard)(x)
    return result.error ? result : success(x as SomeNonPlanParams)
  }

  export const migrate = (x: SomeNonPlanParams): NonPlanParams => {
    if ('v' in x && x.v === currentVersion) return x
    const prev = NonPlanParamsPrev.migrate(x)

    const result: NonPlanParams = {
      v: currentVersion,
      timezone: { type: 'auto' },
      numOfSimulationForMonteCarloSampling:
        prev.numOfSimulationForMonteCarloSampling,
      dev: {
        showDevFeatures: false,
        alwaysShowAllMonths: prev.dev.alwaysShowAllMonths,
        overridePlanResultChartYRange: false,
      },
    }
    assert(!guard(result).error)
    return result
  }
}
