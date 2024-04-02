import {
  MonthRange,
  Person,
  PlanParams,
  LabeledAmountTimed,
  LabeledAmountTimedList,
  assert,
  block,
  currentPlanParamsVersion,
  noCase,
} from '@tpaw/common'
import _ from 'lodash'
import { normalizeGlidePath } from './NormalizeGlidePath'
import {
  PlanParamsNormalized,
  normalizePlanParams,
} from './NormalizePlanParams'
import { NormalizedMonthRange } from './NormalizeLabeledAmountTimedList/NormalizeAmountAndTimingRecurring'
import { NormalizedLabeledAmountTimed } from './NormalizeLabeledAmountTimedList/NormalizeLabeledAmountTimedList'
import jsonpatch from 'fast-json-patch'
import * as Sentry from '@sentry/nextjs'

export const normalizePlanParamsInverse = (
  norm: PlanParamsNormalized,
): PlanParams => {
  const _forMonthRange = (norm: NormalizedMonthRange): MonthRange => {
    switch (norm.type) {
      case 'startAndEnd':
        return {
          type: 'startAndEnd',
          start: norm.start.baseValue,
          end: norm.end.isInThePast
            ? { type: 'inThePast' }
            : norm.end.baseValue,
        }
      case 'startAndDuration':
        return {
          type: 'startAndDuration',
          start: norm.start.baseValue,
          duration: norm.duration.baseValue,
        }
      case 'endAndDuration':
        return {
          type: 'endAndDuration',
          end: norm.end.baseValue,
          duration: norm.duration.baseValue,
        }
      default:
        noCase(norm)
    }
  }

  const _forLabeledAmountTimed = (
    x: NormalizedLabeledAmountTimed,
  ): LabeledAmountTimed => ({
    label: x.label,
    nominal: x.nominal,
    id: x.id,
    sortIndex: x.sortIndex,
    colorIndex: x.colorIndex,
    amountAndTiming: block(() => {
      const norm = x.amountAndTiming
      switch (norm.type) {
        case 'inThePast':
          return { type: 'inThePast' }
        case 'oneTime':
          return {
            type: 'oneTime',
            amount: norm.amount,
            month: norm.month.baseValue,
          }
        case 'recurring':
          return {
            type: 'recurring',
            monthRange: _forMonthRange(norm.monthRange),
            everyXMonths: norm.everyXMonths,
            baseAmount: norm.baseAmount,
            delta: norm.delta,
          }
        default:
          noCase(norm)
      }
    }),
  })

  const _forLabeledAmountTimedList = (
    x: NormalizedLabeledAmountTimed[],
  ): LabeledAmountTimedList =>
    _.fromPairs(x.map((x) => [x.id, _forLabeledAmountTimed(x)]))

  const result: PlanParams = {
    v: currentPlanParamsVersion,
    timestamp: norm.timestamp,
    dialogPositionNominal: norm.dialogPosition.effective,
    people: block(() => {
      const _getPerson = (n: typeof norm.ages.person1): Person => ({
        ages:
          n.retirement.ageIfInFuture === null
            ? {
                type: 'retiredWithNoRetirementDateSpecified',
                monthOfBirth: n.monthOfBirth.baseValue,
                maxAge: n.maxAge.baseValue,
              }
            : {
                type: 'retirementDateSpecified',
                monthOfBirth: n.monthOfBirth.baseValue,
                retirementAge: n.retirement.ageIfInFuture.baseValue,
                maxAge: n.maxAge.baseValue,
              },
      })
      return norm.ages.person2
        ? {
            withPartner: true,
            person1: _getPerson(norm.ages.person1),
            person2: _getPerson(norm.ages.person2),
            withdrawalStart:
              norm.ages.simulationMonths.withdrawalStartMonth.atRetirementOf,
          }
        : {
            withPartner: false,
            person1: _getPerson(norm.ages.person1),
          }
    }),
    wealth: {
      portfolioBalance: norm.wealth.portfolioBalance,
      futureSavings: _forLabeledAmountTimedList(norm.wealth.futureSavings),
      incomeDuringRetirement: _forLabeledAmountTimedList(
        norm.wealth.incomeDuringRetirement,
      ),
    },
    adjustmentsToSpending: {
      extraSpending: {
        essential: _forLabeledAmountTimedList(
          norm.adjustmentsToSpending.extraSpending.essential,
        ),
        discretionary: _forLabeledAmountTimedList(
          norm.adjustmentsToSpending.extraSpending.discretionary,
        ),
      },
      tpawAndSPAW: {
        monthlySpendingCeiling:
          norm.adjustmentsToSpending.tpawAndSPAW.monthlySpendingCeiling,
        monthlySpendingFloor:
          norm.adjustmentsToSpending.tpawAndSPAW.monthlySpendingFloor,
        legacy: {
          total: norm.adjustmentsToSpending.tpawAndSPAW.legacy.total,
          external: _.fromPairs(
            norm.adjustmentsToSpending.tpawAndSPAW.legacy.external.map((x) => [
              x.id,
              x,
            ]),
          ),
        },
      },
    },
    risk: {
      tpaw: norm.risk.tpaw,
      tpawAndSPAW: norm.risk.tpawAndSPAW,
      spaw: norm.risk.spaw,
      spawAndSWR: {
        allocation: normalizeGlidePath.inverse(norm.risk.spawAndSWR.allocation),
      },
      swr: norm.risk.swr,
    },
    advanced: norm.advanced,
    results: norm.results,
  }
  const reNorm = normalizePlanParams(result, norm.nowAs.calendarMonth)
  const diff = jsonpatch.compare(norm, reNorm)
  if (diff.length > 0) {
    Sentry.captureException(
      new Error(`Expected diff to be empty, but got ${JSON.stringify(diff)}`),
    )
  }
  assert(diff.length === 0)
  return result
}
