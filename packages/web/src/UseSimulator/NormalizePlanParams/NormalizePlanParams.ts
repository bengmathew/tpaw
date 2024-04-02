import {
  CalendarMonth,
  DialogPosition,
  Month,
  PlanParams,
  block,
} from '@tpaw/common'
import { PlanParamsHelperFns } from '../PlanParamsHelperFns'
import { normalizeGlidePath } from './NormalizeGlidePath'
import { normalizeLabeledAmountTimedList } from './NormalizeLabeledAmountTimedList/NormalizeLabeledAmountTimedList'
import { normalizeLabeledAmountUntimedList } from './NormalizeLabeledAmountUntimedList'
import { getToMFN, normalizePlanParamsAges } from './NormalizePlanParamsAges'
import { normalize } from 'relay-runtime/lib/store/RelayResponseNormalizer'
import { normalizePlanParamsInverseUnchecked } from './NormalizePlanParamsInverse'
import * as Sentry from '@sentry/nextjs'
import jsonpatch from 'fast-json-patch'

export type PlanParamsNormalized = ReturnType<typeof normalizePlanParams>

export const normalizePlanParams = (
  planParams: PlanParams,
  nowAsCalendarMonth: CalendarMonth,
) => {
  const result = normalizePlanParamsUnchecked(planParams, nowAsCalendarMonth)
  const reNorm = normalizePlanParamsUnchecked(
    normalizePlanParamsInverseUnchecked(result),
    nowAsCalendarMonth,
  )

  const diff = jsonpatch.compare(planParams, reNorm)
  if (diff.length > 0) {
    Sentry.captureMessage(
      `Expected diff to be empty, but got ${JSON.stringify(diff)}`,
    )
  }
  // Don't assert. The assert will happen in the inverse function which is
  // called in updatePlanParams(). This will let the user continue until they
  // make a change.
  // assert(diff.length === 0)
  return result
}
export const normalizePlanParamsUnchecked = (
  planParams: PlanParams,
  nowAsCalendarMonth: CalendarMonth,
) => {
  const ages = normalizePlanParamsAges(planParams.people, nowAsCalendarMonth)
  const toMFN = getToMFN({ nowAs: { calendarMonth: nowAsCalendarMonth }, ages })

  const nowAsMonth: Month = {
    type: 'calendarMonthAsNow',
    monthOfEntry: nowAsCalendarMonth,
  }

  return {
    v: planParams.v,
    timestamp: planParams.timestamp,
    nowAs: {
      calendarMonth: nowAsCalendarMonth,
      month: nowAsMonth,
    },
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
        toMFN,
        ages,
      ),
      incomeDuringRetirement: normalizeLabeledAmountTimedList(
        planParams.wealth.incomeDuringRetirement,
        ages.validMonthRangesAsMFN.incomeDuringRetirement,
        toMFN,
        ages,
      ),
    },

    adjustmentsToSpending: {
      extraSpending: {
        essential: normalizeLabeledAmountTimedList(
          planParams.adjustmentsToSpending.extraSpending.essential,
          ages.validMonthRangesAsMFN.extraSpending,
          toMFN,
          ages,
        ),
        discretionary: normalizeLabeledAmountTimedList(
          planParams.adjustmentsToSpending.extraSpending.discretionary,
          ages.validMonthRangesAsMFN.extraSpending,
          toMFN,
          ages,
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
          toMFN,
          ages,
        ),
      },
      swr: planParams.risk.swr,
    },
    advanced: planParams.advanced,
    results: planParams.results,
  }
}
