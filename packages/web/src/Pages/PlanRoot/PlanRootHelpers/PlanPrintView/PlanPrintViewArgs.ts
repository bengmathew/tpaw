import {
  NonPlanParams,
  PlanParams
} from '@tpaw/common'
import { DailyMarketSeriesSrc } from '../../../../Simulator/SimulateOnServer/SimulateOnServer'

type _PageSize_NoDefault = Exclude<
  NonPlanParams['pdfReportSettings']['pageSize'],
  'default'
>

export type PlanPrintViewArgs = {
  fixed: {
    planLabel: string | null
    datingInfo:
      | {
          isDatedPlan: true
          simulationTimestamp: number
          ianaTimezoneName: string
        }
      | { isDatedPlan: false; timestampForMarketData: number }
    // We want the estimate to be resolved to an amount for printing and don't
    // surface any estimation info. This is needed for because we may add a link
    // to the plan in the PDF and links don't support current portfolio balance
    // estimation.
    currentPortfolioBalanceAmount: number
    planParams: PlanParams
    dailyMarketSeriesSrc: DailyMarketSeriesSrc
    percentiles: { low: number; mid: number; high: number }
    numOfSimulationForMonteCarloSampling: number
    randomSeed: number
  }

  settings:
    | {
        isServerSidePrint: false
        pageSize: _PageSize_NoDefault
        shouldEmbedLink: boolean
        alwaysShowAllMonths: boolean
      }
    | {
        isServerSidePrint: true
        pageSize: _PageSize_NoDefault
        linkToEmbed: string | null
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
