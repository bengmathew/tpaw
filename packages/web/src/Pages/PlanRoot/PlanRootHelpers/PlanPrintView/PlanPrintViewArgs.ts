import {
  DataForMarketBasedPlanParamValues,
  NonPlanParams,
  PlanParams,
} from '@tpaw/common'

type _PageSize_NoDefault = Exclude<
  NonPlanParams['pdfReportSettings']['pageSize'],
  'default'
>

export type PlanPrintViewArgs = {
  fixed: {
    planLabel: string | null
    planParams: PlanParams
    marketData: DataForMarketBasedPlanParamValues
    numOfSimulationForMonteCarloSampling: number
    ianaTimezoneName: string
    randomSeed: number
  }

  settings:
    | {
        isServerSidePrint: false
        pageSize: _PageSize_NoDefault
        embeddedLinkType: Exclude<
          NonPlanParams['pdfReportSettings']['embeddedLinkType'],
          'default'
        >
        alwaysShowAllMonths: boolean
      }
    | {
        isServerSidePrint: true
        pageSize: _PageSize_NoDefault
        linkToEmbed: string
        alwaysShowAllMonths: boolean
      }
}

export type PlanPrintViewSettingsClientSide = Extract<
  PlanPrintViewArgs['settings'],
  { isServerSidePrint: false }
>
export type PlanPrintViewSettingsSeverSide = Extract<
  PlanPrintViewArgs['settings'],
  { isServerSidePrint: true }
>

export type PlanPrintViewSettingsControlledClientSide = Omit<
  PlanPrintViewSettingsClientSide,
  'alwaysShowAllMonths' | 'isServerSidePrint'
>

export type PlanPrintViewArgsServerSide = {
  fixed: PlanPrintViewArgs['fixed']
  settings: PlanPrintViewSettingsSeverSide
}
