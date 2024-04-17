import {
  CalendarMonth,
  DialogPosition,
  Month,
  PlanParams,
  assert,
  block,
  fGet,
  letIn,
} from '@tpaw/common'
import { PlanParamsHelperFns } from '../PlanParamsHelperFns'
import { normalizeGlidePath } from './NormalizeGlidePath'
import { normalizeLabeledAmountTimedList } from './NormalizeLabeledAmountTimedList/NormalizeLabeledAmountTimedList'
import { normalizeLabeledAmountUntimedList } from './NormalizeLabeledAmountUntimedList'
import { getMonthToMFN, normalizePlanParamsAges } from './NormalizePlanParamsAges'
import { normalizePlanParamsInverse } from './NormalizePlanParamsInverse'
import _ from 'lodash'

export type PlanParamsNormalized = ReturnType<typeof normalizePlanParams>

export const normalizePlanParams = (
  planParams: PlanParams,
  nowAs: {
    timestamp: number
    // Cannot be null if dated plan.
    calendarMonth: CalendarMonth | null
  },
) => {
  const norm = normalizePlanParamsUnchecked(planParams, nowAs)
  // Soft check inverse.
  const deNorm = normalizePlanParamsInverse(norm, 'soft')
  if (!planParams.datingInfo.isDated) assert(_.isEqual(deNorm, planParams))
  return norm
}

type NormalizedDatingInfo =
  | {
      isDated: true
      nowAsCalendarMonth: CalendarMonth
      nowAsTimestamp: number
    }
  | {
      isDated: false
      nowAsCalendarMonth: null
      timestampForMarketData: number
      nowAsTimestampNominal: number
    }
export const normalizePlanParamsUnchecked = (
  planParams: PlanParams,
  nowAs: {
    timestamp: number
    calendarMonth: CalendarMonth | null
  },
) => {
  const datingInfo: NormalizedDatingInfo = planParams.datingInfo.isDated
    ? ({
        isDated: true,
        nowAsTimestamp: nowAs.timestamp,
        nowAsCalendarMonth: fGet(nowAs.calendarMonth),
      } as const)
    : ({
        isDated: false,
        nowAsCalendarMonth: null,
        timestampForMarketData: planParams.datingInfo.timestampForMarketData,
        nowAsTimestampNominal: nowAs.timestamp,
      } as const)
  const { nowAsCalendarMonth } = datingInfo
  const ages = normalizePlanParamsAges(planParams.people, nowAsCalendarMonth)
  const monthToMFN = getMonthToMFN(nowAsCalendarMonth, ages)
  return {
    v: planParams.v,
    timestamp: planParams.timestamp,
    datingInfo,
    dialogPosition: block(() => {
      const getEffective = (
        curr: DialogPosition,
        isFutureSavingsAllowed: boolean,
      ) =>
        curr === 'future-savings' && !isFutureSavingsAllowed
          ? PlanParamsHelperFns.getNextDialogPosition('future-savings')
          : curr

      const effective = getEffective(
        planParams.dialogPositionNominal,
        ages.validMonthRangesAsMFN.futureSavings !== null,
      )

      const next =
        effective === 'done'
          ? 'done'
          : getEffective(
              PlanParamsHelperFns.getNextDialogPosition(effective),
              ages.validMonthRangesAsMFN.futureSavings !== null,
            )

      return { effective, next }
    }),
    ages,
    wealth: {
      portfolioBalance: planParams.wealth.portfolioBalance,
      futureSavings: normalizeLabeledAmountTimedList(
        planParams.wealth.futureSavings,
        ages.validMonthRangesAsMFN.futureSavings,
        monthToMFN,
        ages,
        nowAsCalendarMonth,
      ),
      incomeDuringRetirement: normalizeLabeledAmountTimedList(
        planParams.wealth.incomeDuringRetirement,
        ages.validMonthRangesAsMFN.incomeDuringRetirement,
        monthToMFN,
        ages,
        nowAsCalendarMonth,
      ),
    },

    adjustmentsToSpending: {
      extraSpending: {
        essential: normalizeLabeledAmountTimedList(
          planParams.adjustmentsToSpending.extraSpending.essential,
          ages.validMonthRangesAsMFN.extraSpending,
          monthToMFN,
          ages,
          nowAsCalendarMonth,
        ),
        discretionary: normalizeLabeledAmountTimedList(
          planParams.adjustmentsToSpending.extraSpending.discretionary,
          ages.validMonthRangesAsMFN.extraSpending,
          monthToMFN,
          ages,
          nowAsCalendarMonth,
        ),
      },
      tpawAndSPAW: {
        monthlySpendingCeiling:
          planParams.adjustmentsToSpending.tpawAndSPAW.monthlySpendingCeiling,
        monthlySpendingFloor:
          planParams.adjustmentsToSpending.tpawAndSPAW.monthlySpendingFloor,
        legacy: {
          total: planParams.adjustmentsToSpending.tpawAndSPAW.legacy.total,
          external: normalizeLabeledAmountUntimedList(
            planParams.adjustmentsToSpending.tpawAndSPAW.legacy.external,
          ),
        },
      },
    },

    risk: {
      tpaw: planParams.risk.tpaw,
      tpawAndSPAW: planParams.risk.tpawAndSPAW,
      spaw: planParams.risk.spaw,
      spawAndSWR: {
        allocation: normalizeGlidePath(
          planParams.risk.spawAndSWR.allocation,
          monthToMFN,
          ages,
          nowAsCalendarMonth,
        ),
      },
      swr: planParams.risk.swr,
    },
    advanced: planParams.advanced,
    results: planParams.results,
  }
}
