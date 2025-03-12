
import _ from 'lodash'
import {
  NormalizedMonthNotInThePast,
  normalizedMonthRangeCheckAndSquishRangeForAge,
} from './NormalizeLabeledAmountTimedList/NormalizedMonth'
import { NormalizedAges, MonthToMFN } from './NormalizeAges'
import { CalendarMonthFns } from '../../../Misc/CalendarMonthFns'
import { letIn, fGet, block, linearFnFomPoints } from '../../../Utils'
import { GlidePath, CalendarMonth } from '../PlanParams'

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
    const month = normalizedMonthRangeCheckAndSquishRangeForAge(x.month, ages)
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
      .map((x) => ({ ...x, month: { ...x.month, isInThePast: false } })),
    end: { stocks: orig.end.stocks },
    atOrPastEnd: atOrPastEndStage1.map((x) => {
      const month = normalizedMonthRangeCheckAndSquishRangeForAge(x.month, ages)
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
