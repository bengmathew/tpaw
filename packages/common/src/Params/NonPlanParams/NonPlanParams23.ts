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
import { NonPlanParams22 as NonPlanParamsPrev } from './Old/NonPlanParams22'
export namespace NonPlanParams23 {
  export const currentVersion = 23 as const
  export type NonPlanParams = {
    v: typeof currentVersion
    timezone: { type: 'auto' } | { type: 'manual'; ianaTimezoneName: string }
    numOfSimulationForMonteCarloSampling: number
    dev: {
      showSyncStatus: boolean
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
      showSyncStatus: boolean,
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
      timezone: prev.timezone,
      numOfSimulationForMonteCarloSampling:
        prev.numOfSimulationForMonteCarloSampling,
      dev: {
        showSyncStatus: false,
        showDevFeatures: prev.dev.showDevFeatures,
        alwaysShowAllMonths: prev.dev.alwaysShowAllMonths,
        overridePlanResultChartYRange: prev.dev.overridePlanResultChartYRange,
      },
    }
    assert(!guard(result).error)
    return result
  }
}
