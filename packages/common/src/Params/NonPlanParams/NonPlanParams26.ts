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
import { assert, block, letIn, noCase } from '../../Utils'
import { NonPlanParams25 as NonPlanParamsPrev } from './Old/NonPlanParams25'
export namespace NonPlanParams26 {
  // Just to re-emphasize that currentVersion has the const type.
  export const currentVersion = 26 as number as 26
  export type NonPlanParams = {
    v: typeof currentVersion
    timestamp: number
    timezone: { type: 'auto' } | { type: 'manual'; ianaTimezoneName: string }
    numOfSimulationForMonteCarloSampling: number
    showOfflinePlansMenuSection: boolean
    pdfReportSettings: {
      pageSize: 'A4' | 'Letter' | 'default'
      shouldEmbedLink: 'auto' | 'yes' | 'no'
    }
    dev: {
      showSyncStatus: boolean
      showDevFeatures: boolean
      alwaysShowAllMonths: boolean
      overridePlanResultChartYRange: false | { start: number; end: number }
    }
  }

  export const fns = block(() => {
    const resolvePDFReportSettingsDefaults = {
      pageSize: (pageSize: NonPlanParams['pdfReportSettings']['pageSize']) =>
        pageSize === 'default'
          ? letIn(new Intl.Locale(navigator.language).region, (region) =>
              region === 'US' || region === 'CA'
                ? ('A4' as const)
                : ('Letter' as const),
            )
          : pageSize,
      getShouldEmbedLink: (
        shouldEmbedLink: NonPlanParams['pdfReportSettings']['shouldEmbedLink'],
        isLoggedIn: boolean,
      ) =>
        shouldEmbedLink === 'auto'
          ? isLoggedIn
          : shouldEmbedLink === 'yes'
            ? true
            : shouldEmbedLink === 'no'
              ? false
              : noCase(shouldEmbedLink),
    }

    return { resolvePDFReportSettingsDefaults }
  })

  // ----------- GUARD  ---------//
  export const guard: JSONGuard<NonPlanParams> = object({
    v: constant(currentVersion),
    timestamp: number,
    timezone: union(
      object({ type: constant('auto') }),
      object({
        type: constant('manual'),
        ianaTimezoneName: Guards.ianaTimezoneName,
      }),
    ),
    showOfflinePlansMenuSection: boolean,
    numOfSimulationForMonteCarloSampling: chain(number, integer, gt(0)),
    pdfReportSettings: object({
      pageSize: union(constant('A4'), constant('Letter'), constant('default')),
      shouldEmbedLink: union(constant('auto'), constant('yes'), constant('no')),
    }),
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
      ...prev,
      v: currentVersion,
      showOfflinePlansMenuSection: false,
    }
    assert(!guard(result).error)
    return result
  }
}
