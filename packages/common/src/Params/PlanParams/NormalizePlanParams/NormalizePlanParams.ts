
import { getNYZonedTime } from '../../../Misc/GetZonedTimeFns'
import { fGet, block } from '../../../Utils'
import { PlanParams, CalendarDay, DialogPosition } from '../PlanParams'
import { PlanParamsHelperFns } from '../PlanParamsHelperFns'
import { getMonthToMFN, normalizeAges } from './NormalizeAges'
import { normalizeGlidePath } from './NormalizeGlidePath'
import { normalizeLabeledAmountTimedList } from './NormalizeLabeledAmountTimedList/NormalizeLabeledAmountTimedList'
import { normalizeLabeledAmountUntimedList } from './NormalizeLabeledAmountUntimedList'
import { normalizePlanParamsInverse } from './NormalizePlanParamsInverse'
import _ from 'lodash'

export type PlanParamsNormalized = ReturnType<typeof normalizePlanParams>

export const normalizePlanParams = (
  planParams: PlanParams,
  nowAs: {
    timestamp: number
    // Cannot be null if dated plan.
    calendarDay: CalendarDay | null
  },
) => {
  const norm = normalizePlanParamsUnchecked(planParams, nowAs)
  normalizePlanParamsInverse(norm)
  return norm
}

export type NormalizedDatingInfo =
  | {
      isDated: true
      nowAsCalendarDay: CalendarDay
      timestampForMarketData: number
      nowAsTimestamp: number
    }
  | {
      isDated: false
      nowAsCalendarDay: null
      timestampForMarketData: number
      marketDataAsOfEndOfDayInNY: CalendarDay
      nowAsTimestampNominal: number
    }
export const normalizePlanParamsUnchecked = (
  planParams: PlanParams,
  nowAs: {
    timestamp: number
    calendarDay: CalendarDay | null
  },
) => {
  const datingInfo: NormalizedDatingInfo = planParams.datingInfo.isDated
    ? ({
        isDated: true,
        nowAsTimestamp: nowAs.timestamp,
        nowAsCalendarDay: fGet(nowAs.calendarDay),
        timestampForMarketData: nowAs.timestamp,
      } as const)
    : ({
        isDated: false,
        nowAsCalendarDay: null,
        marketDataAsOfEndOfDayInNY:
          planParams.datingInfo.marketDataAsOfEndOfDayInNY,
        timestampForMarketData: getNYZonedTime
          .fromObject(planParams.datingInfo.marketDataAsOfEndOfDayInNY)
          .endOf('day')
          .toMillis(),
        nowAsTimestampNominal: nowAs.timestamp,
      } as const)
  const { nowAsCalendarDay } = datingInfo
  const ages = normalizeAges(planParams.people, nowAsCalendarDay)
  const monthToMFN = getMonthToMFN(nowAsCalendarDay, ages)
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
        nowAsCalendarDay,
      ),
      incomeDuringRetirement: normalizeLabeledAmountTimedList(
        planParams.wealth.incomeDuringRetirement,
        ages.validMonthRangesAsMFN.incomeDuringRetirement,
        monthToMFN,
        ages,
        nowAsCalendarDay,
      ),
    },

    adjustmentsToSpending: {
      extraSpending: {
        essential: normalizeLabeledAmountTimedList(
          planParams.adjustmentsToSpending.extraSpending.essential,
          ages.validMonthRangesAsMFN.extraSpending,
          monthToMFN,
          ages,
          nowAsCalendarDay,
        ),
        discretionary: normalizeLabeledAmountTimedList(
          planParams.adjustmentsToSpending.extraSpending.discretionary,
          ages.validMonthRangesAsMFN.extraSpending,
          monthToMFN,
          ages,
          nowAsCalendarDay,
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
          nowAsCalendarDay,
        ),
      },
      swr: planParams.risk.swr,
    },
    advanced: planParams.advanced,
    results: planParams.results,
  }
}
