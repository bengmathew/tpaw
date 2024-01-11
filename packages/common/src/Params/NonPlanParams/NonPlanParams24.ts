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
import { assert, block, letIn } from '../../Utils'
import { NonPlanParams23 as NonPlanParamsPrev } from './Old/NonPlanParams23'
export namespace NonPlanParams24 {
  export const currentVersion = 24 as const
  export type NonPlanParams = {
    v: typeof currentVersion
    timestamp: number
    timezone: { type: 'auto' } | { type: 'manual'; ianaTimezoneName: string }
    numOfSimulationForMonteCarloSampling: number
    pdfReportSettings: {
      pageSize: 'A4' | 'Letter' | 'default'
      embeddedLinkType: 'short' | 'long' | 'default'
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
      embeddedLinkType: (
        embeddedLinkType: NonPlanParams['pdfReportSettings']['embeddedLinkType'],
        isLoggedIn: boolean,
      ) =>
        embeddedLinkType === 'default'
          ? isLoggedIn
            ? 'short'
            : 'long'
          : embeddedLinkType,
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
    numOfSimulationForMonteCarloSampling: chain(number, integer, gt(0)),
    pdfReportSettings: object({
      pageSize: union(constant('A4'), constant('Letter'), constant('default')),
      embeddedLinkType: union(
        constant('short'),
        constant('long'),
        constant('default'),
      ),
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
      timestamp: 1704931412864,
      pdfReportSettings: {
        pageSize: 'default',
        embeddedLinkType: 'default',
      },
    }
    assert(!guard(result).error)
    return result
  }
}
