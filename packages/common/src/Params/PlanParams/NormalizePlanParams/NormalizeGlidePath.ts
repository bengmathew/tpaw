import _ from 'lodash'
import {
  NormalizedMonthNotInThePast,
  normalizedMonthRangeCheckAndSquishRangeForAge,
} from './NormalizeLabeledAmountTimedList/NormalizedMonth'
import { NormalizedAges, MonthToMFN } from './NormalizeAges'
import { CalendarMonthFns } from '../../../Misc/CalendarMonthFns'
import { letIn, fGet, block, linearFnFomPoints, assert } from '../../../Utils'
import { GlidePath, CalendarMonth, Month } from '../PlanParams'

export type NormalizedGlidePathEntry = {
  id: string
  ignore: boolean
  month: NormalizedMonthNotInThePast
  stocks: number
  indexToSortByAdded: number
}

export type NormalizedGlidePath = {
  now: { stocks: number }
  intermediate: NormalizedGlidePathEntry[]
  end: { stocks: number }
  atOrPastEnd: NormalizedGlidePathEntry[]
}

export const normalizeGlidePath = (
  orig: GlidePath,
  monthToMFN: MonthToMFN,
  ages: NormalizedAges,
  nowAsCalendarDay: CalendarMonth | null,
): NormalizedGlidePath => {
  const { lastMonthAsMFN } = ages.simulationMonths
  const validRangeAsMFN = letIn({ start: 1, end: lastMonthAsMFN - 1 }, (x) => ({
    includingLocalConstraints: x,
    excludingLocalConstraints: x,
  }))
  const nowMonth: Extract<Month, { type: 'now' }> = {
    type: 'now',
    monthOfEntry: nowAsCalendarDay
      ? { isDatedPlan: true, calendarMonth: nowAsCalendarDay }
      : { isDatedPlan: false },
  }
  const preNormStartAsMFN = orig.start.month.monthOfEntry.isDatedPlan
    ? CalendarMonthFns.getToMFN(fGet(nowAsCalendarDay))(
        orig.start.month.monthOfEntry.calendarMonth,
      )
    : 0
  const [beforeEndStage1, atOrPastEndStage1] = _.partition(
    _.values(orig.intermediate)
      .map((x) => ({
        id: x.id,
        month: {
          asMFN: monthToMFN.pastNotElided(x.month),
          baseValue: x.month,
          validRangeAsMFN,
        },
        stocks: x.stocks,
        indexToSortByAdded: x.indexToSortByAdded,
      }))
      .sort((a, b) => a.indexToSortByAdded - b.indexToSortByAdded)
      .sort((a, b) => a.month.asMFN - b.month.asMFN)
      .filter((x) => x.month.asMFN > preNormStartAsMFN),
    (x) => x.month.asMFN < lastMonthAsMFN,
  )

  const intermediateBeforeElidingPast = beforeEndStage1.map((x, i) => {
    const duplicate =
      i > 0 && x.month.asMFN === fGet(beforeEndStage1[i - 1]).month.asMFN
    const month = normalizedMonthRangeCheckAndSquishRangeForAge(
      x.month,
      ages,
      nowMonth,
    )
    return {
      ...x,
      ignore: duplicate,
      month: duplicate
        ? { ...month, errorMsg: normalizedGlidePathErrorMsg.duplicate }
        : month,
    }
  })

  const stocksNow = block(() => {
    const preNormSimplePath = [
      { mfn: preNormStartAsMFN, stocks: orig.start.stocks },
      ...intermediateBeforeElidingPast
        .filter((x) => !x.ignore)
        .map((x) => ({ mfn: x.month.asMFN, stocks: x.stocks })),
      { mfn: lastMonthAsMFN, stocks: orig.end.stocks },
    ]

    // It is possible that the start might not be mfn === 0 . This can happen
    // if the timezone reported a particular month when start was updated, but
    // then user switched to a different timezone (an earlier one) and the new
    // timezone reports the previous month for the evaluation time. The solution
    // in this case is to interpret start as being the current month in the new
    // timezone.
    const first = fGet(_.first(preNormSimplePath))
    if (first.mfn > 0) {
      assert(first.mfn === 1)
      first.mfn = 0
    }

    const lastAtOrBeforeNow = fGet(
      _.findLast(preNormSimplePath, (x) => x.mfn <= 0),
    )
    if (lastAtOrBeforeNow.mfn === 0) return lastAtOrBeforeNow.stocks
    const firstAfterNow = fGet(_.find(preNormSimplePath, (x) => x.mfn > 0))
    return _.round(
      linearFnFomPoints(
        lastAtOrBeforeNow.mfn,
        lastAtOrBeforeNow.stocks,
        firstAfterNow.mfn,
        firstAfterNow.stocks,
      )(0),
      2,
    )
  })

  return {
    now: { stocks: stocksNow },

    intermediate: intermediateBeforeElidingPast
      .filter((x) => x.month.asMFN > 0)
      // TODO: this filter does not seem to be doing anything because
      // month is already NormalizedMonthNotInThePast?
      .map((x) => ({ ...x, month: { ...x.month, isInThePast: false } })),
    end: { stocks: orig.end.stocks },
    atOrPastEnd: atOrPastEndStage1.map((x) => {
      const month = normalizedMonthRangeCheckAndSquishRangeForAge(x.month, ages, nowMonth)
      return {
        ...x,
        ignore: true,
        month: {
          ...month,
          errorMsg: normalizedGlidePathErrorMsg.atOrPastEnd,
        },
      }
    }),
  }
}

normalizeGlidePath.inverse = (
  n: NormalizedGlidePath,
  nowAsCalendarDay: CalendarMonth | null,
): GlidePath => ({
  start: {
    month: {
      type: 'now',
      monthOfEntry: nowAsCalendarDay
        ? {
            isDatedPlan: true,
            calendarMonth: {
              month: nowAsCalendarDay.month,
              year: nowAsCalendarDay.year,
            },
          }
        : { isDatedPlan: false },
    },
    stocks: n.now.stocks,
  },
  intermediate: _.fromPairs(
    [...n.intermediate, ...n.atOrPastEnd].map((x) => [
      x.id,
      {
        id: x.id,
        month: x.month.baseValue,
        stocks: x.stocks,
        indexToSortByAdded: x.indexToSortByAdded,
      },
    ]),
  ),
  end: { stocks: n.end.stocks },
})

export const normalizedGlidePathErrorMsg = {
  duplicate: 'Duplicate entry for this month',
  atOrPastEnd: 'Past end',
}
