import { CalendarDay, CalendarMonth, NonPlanParams, PlanParams } from '@tpaw/common'
import * as Rust from '@tpaw/simulator'
import { CurrentPortfolioBalance } from '../CurrentPortfolioBalance'
import { isDate } from 'lodash'

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
          nowAsTimestamp: number
          nowAsCalendarDay: CalendarDay
        }
      | { isDatedPlan: false; timestampForMarketData: number }
    // We want the estimate to be resolved to an amount for printing and don't
    // surface any estimation info. This is needed for because we add a link to
    // the plan in the PDF and links don't support current portfolio balance
    // estimation.
    currentPortfolioBalanceAmount: number
    planParams: PlanParams
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
