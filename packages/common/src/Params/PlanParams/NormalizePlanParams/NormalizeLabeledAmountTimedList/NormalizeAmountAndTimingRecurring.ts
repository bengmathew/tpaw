import _, { isDate } from 'lodash'
import {
  NormalizedAges,
  MonthToMFN,
  getFromMFNToNumericAge,
} from '../NormalizeAges'
import {
  NormalizedMonthInThePast,
  NormalizedMonthNotInThePast,
  getNormalizedMonthNotInThePast,
  normalizedMonthErrorMsg,
} from './NormalizedMonth'
import { SimpleRange } from '../../../../Misc/SimpleRange'
import {
  PickType,
  block,
  assert,
  noCase,
  fGet,
  assertFalse,
} from '../../../../Utils'
import { LabeledAmountTimed, CalendarMonth, Month } from '../../PlanParams'
import { CalendarMonthFns } from '../../../../Misc/CalendarMonthFns'
import { Record } from '../../../../Misc/Record'
import { InMonthsFns } from '../../../../Misc/InMonthsFns'

type _Orig = PickType<LabeledAmountTimed['amountAndTiming'], 'recurring'>

export type NormalizedDurationNotInThePast = {
  type: 'normalizedDuration'
  isInThePast: false
  asMFN: number
  baseValue: { inMonths: number }
  validRangeInMonths: { includingLocalConstraints: SimpleRange | null }
  errorMsg: string | null
}

const _getNormalizedDuration = (
  type: 'startAndDuration' | 'endAndDuration',
  asMFN: number,
  baseValue: { inMonths: number },
  validRangeAsMFN: { includingLocalConstraints: SimpleRange | null },
): NormalizedDurationNotInThePast => {
  const { includingLocalConstraints } = validRangeAsMFN
  return {
    type: 'normalizedDuration',
    isInThePast: false,
    asMFN,
    baseValue,
    validRangeInMonths: {
      includingLocalConstraints: includingLocalConstraints
        ? block(() => {
            switch (type) {
              case 'startAndDuration':
                return Record.mapValues(
                  includingLocalConstraints,
                  InMonthsFns.getFromMFN(baseValue, asMFN, 1),
                )
              case 'endAndDuration': {
                const fromMFN = InMonthsFns.getFromMFN(baseValue, asMFN, -1)
                // Order is flipped because slope is -1.
                return {
                  start: fromMFN(includingLocalConstraints.end),
                  end: fromMFN(includingLocalConstraints.start),
                }
              }
            }
          })
        : null,
    },
    errorMsg: SimpleRange.Closed.isIn(asMFN, includingLocalConstraints)
      ? null
      : normalizedMonthErrorMsg.outOfRange,
  }
}

export type NormalizedMonthRange =
  | {
      type: 'startAndEnd'
      start: NormalizedMonthNotInThePast
      end: NormalizedMonthNotInThePast | NormalizedMonthInThePast
      validRangeAsMFN: SimpleRange
    }
  | {
      type: 'startAndDuration'
      start: NormalizedMonthNotInThePast
      duration: NormalizedDurationNotInThePast
      validRangeAsMFN: SimpleRange
    }
  | {
      type: 'endAndDuration'
      end: NormalizedMonthNotInThePast
      duration: NormalizedDurationNotInThePast
      validRangeAsMFN: SimpleRange
    }

export type NormalizedAmountAndTimingRecurring = {
  type: 'recurring'
  monthRange: NormalizedMonthRange
  everyXMonths: _Orig['everyXMonths']
  baseAmount: _Orig['baseAmount']
  delta: _Orig['delta']
}

export const normalizeAmountAndTimingRecurring = (
  {
    monthRange,
    everyXMonths,
    delta,
    baseAmount,
  }: PickType<LabeledAmountTimed['amountAndTiming'], 'recurring'>,
  validRangeAsMFN: SimpleRange,
  monthToMFN: MonthToMFN,
  ages: NormalizedAges,
  nowAsCalendarMonth: CalendarMonth | null,
): NormalizedAmountAndTimingRecurring | null => {
  const result = (monthRange: NormalizedMonthRange, baseAmount: number) => ({
    type: 'recurring' as const,
    monthRange,
    everyXMonths,
    baseAmount,
    delta,
  })

  const mfnTo = nowAsCalendarMonth
    ? ({
        isDatedPlan: true,
        calendarMonth: CalendarMonthFns.getFromMFN(nowAsCalendarMonth),
      } as const)
    : ({
        isDatedPlan: false,
        numericAge: getFromMFNToNumericAge({ ages }),
      } as const)

  switch (monthRange.type) {
    case 'startAndEnd': {
      const inputStartAsMFNPastNotElided = monthToMFN.pastNotElided(
        monthRange.start,
      )
      const inputEndAsMFNPastElided =
        monthRange.end.type === 'inThePast'
          ? 'inThePast'
          : monthToMFN.pastElided(monthRange.end)
      if (
        inputStartAsMFNPastNotElided < 0 &&
        inputEndAsMFNPastElided === 'inThePast'
      ) {
        return null
      } else {
        const startInfo = _stepStartToCurrent(
          {
            asMFNPastNotElided: inputStartAsMFNPastNotElided,
            value: monthRange.start,
          },
          inputEndAsMFNPastElided,
          baseAmount,
          everyXMonths,
          delta,
          mfnTo,
        )
        if (!startInfo) return null
        const start = getNormalizedMonthNotInThePast(
          startInfo.month.asMFN,
          startInfo.month.value,
          nowAsCalendarMonth,
          {
            includingLocalConstraints: validRangeAsMFN,
            excludingLocalConstraints: validRangeAsMFN,
          },
          { type: 'rangeCheckAndSquishRangeForAge', ages },
        )

        const end = block(
          (): NormalizedMonthNotInThePast | NormalizedMonthInThePast => {
            const validRangeAsMFNForEnd = _squashValidRangeAsMFNGivenStart(
              start,
              validRangeAsMFN,
            )
            if (inputEndAsMFNPastElided === 'inThePast')
              return {
                type: 'normalizedMonth',
                isInThePast: true,
                validRangeAsMFN: {
                  includingLocalConstraints: validRangeAsMFNForEnd,
                  excludingLocalConstraints: validRangeAsMFN,
                },
                errorMsg: normalizedMonthErrorMsg.outOfRange,
              }
            assert(monthRange.end.type !== 'inThePast')
            return getNormalizedMonthNotInThePast(
              inputEndAsMFNPastElided,
              monthRange.end,
              nowAsCalendarMonth,
              {
                includingLocalConstraints: validRangeAsMFNForEnd,
                excludingLocalConstraints: validRangeAsMFN,
              },
              { type: 'rangeCheckAndSquishRangeForAge', ages },
            )
          },
        )
        return result(
          { type: 'startAndEnd', start, end, validRangeAsMFN },
          startInfo.baseAmount,
        )
      }
    }
    case 'startAndDuration': {
      const inputStartAsMFNPastNotElided = monthToMFN.pastNotElided(
        monthRange.start,
      )
      const inputEndAsMFNPastNotElided =
        inputStartAsMFNPastNotElided + monthRange.duration.inMonths - 1
      if (inputEndAsMFNPastNotElided < 0) return null
      const inputEndAsMFNNotInPast = inputEndAsMFNPastNotElided
      const startInfo = _stepStartToCurrent(
        {
          asMFNPastNotElided: inputStartAsMFNPastNotElided,
          value: monthRange.start,
        },
        inputEndAsMFNNotInPast,
        baseAmount,
        everyXMonths,
        delta,
        mfnTo,
      )
      if (!startInfo) return null
      const start = getNormalizedMonthNotInThePast(
        startInfo.month.asMFN,
        startInfo.month.value,
        nowAsCalendarMonth,
        {
          includingLocalConstraints: validRangeAsMFN,
          excludingLocalConstraints: validRangeAsMFN,
        },
        { type: 'rangeCheckAndSquishRangeForAge', ages },
      )

      assert(inputEndAsMFNNotInPast >= start.asMFN)
      const duration = _getNormalizedDuration(
        monthRange.type,
        inputEndAsMFNNotInPast,
        { inMonths: inputEndAsMFNNotInPast - start.asMFN + 1 },
        {
          includingLocalConstraints: _squashValidRangeAsMFNGivenStart(
            start,
            validRangeAsMFN,
          ),
        },
      )
      return result(
        {
          type: 'startAndDuration',
          start,
          duration,
          validRangeAsMFN,
        },
        startInfo.baseAmount,
      )
    }

    case 'endAndDuration': {
      const inputEndAsMFNPastElided = monthToMFN.pastElided(monthRange.end)
      if (inputEndAsMFNPastElided === 'inThePast') return null
      const inputEndAsMFNNotInPast = inputEndAsMFNPastElided

      const inputStartAsMFNPastNotElided =
        inputEndAsMFNNotInPast - monthRange.duration.inMonths + 1

      const end = getNormalizedMonthNotInThePast(
        inputEndAsMFNNotInPast,
        monthRange.end,
        nowAsCalendarMonth,
        {
          includingLocalConstraints: validRangeAsMFN,
          excludingLocalConstraints: validRangeAsMFN,
        },
        { type: 'rangeCheckAndSquishRangeForAge', ages },
      )
      const startAsMFNNotInPast = Math.max(0, inputStartAsMFNPastNotElided)
      const duration = _getNormalizedDuration(
        monthRange.type,
        startAsMFNNotInPast,
        { inMonths: end.asMFN - startAsMFNNotInPast + 1 },
        {
          includingLocalConstraints: _squashValidRangeAsMFNGivenEnd(
            end,
            validRangeAsMFN,
          ),
        },
      )

      return result(
        {
          type: 'endAndDuration',
          end,
          duration,
          validRangeAsMFN,
        },
        baseAmount,
      )
    }
    default:
      noCase(monthRange)
  }
}

const _squashValidRangeAsMFNGivenStart = (
  normalizedStart: { asMFN: number },
  validRangeAsMFN: SimpleRange,
) =>
  SimpleRange.Closed.intersection(
    SimpleRange.Closed.create(normalizedStart.asMFN, validRangeAsMFN.end),
    validRangeAsMFN,
  )

const _squashValidRangeAsMFNGivenEnd = (
  normalizedEnd: { asMFN: number },
  validRangeAsMFN: SimpleRange,
) =>
  SimpleRange.Closed.intersection(
    SimpleRange.Closed.create(validRangeAsMFN.start, normalizedEnd.asMFN),
    validRangeAsMFN,
  )

const _stepStartToCurrent = (
  startMonth: { asMFNPastNotElided: number; value: Month },
  endMonthAsMFNPastElided: number | 'inThePast',
  baseAmount: number,
  everyXMonths: _Orig['everyXMonths'],
  delta: _Orig['delta'],
  mfnTo:
    | {
        isDatedPlan: true
        calendarMonth: (mfn: number) => CalendarMonth
      }
    | {
        isDatedPlan: false
        numericAge: ReturnType<typeof getFromMFNToNumericAge>
      },
): { month: { asMFN: number; value: Month }; baseAmount: number } | null => {
  const unchanged = {
    month: { asMFN: startMonth.asMFNPastNotElided, value: startMonth.value },
    baseAmount,
  }
  if (startMonth.asMFNPastNotElided >= 0) return unchanged

  // Undated plans cannot have been in the past.
  assert(endMonthAsMFNPastElided !== 'inThePast')
  const endMonthAsMFNNotInPast = endMonthAsMFNPastElided

  const startSteppedToCurrent = _stepRecurringAmountToCurrent(
    { monthAsMFN: startMonth.asMFNPastNotElided, value: baseAmount },
    everyXMonths,
    delta,
  )
  if (startSteppedToCurrent.monthAsMFN > endMonthAsMFNNotInPast) return null

  return {
    month: {
      asMFN:startSteppedToCurrent.monthAsMFN,
      value: block((): Month => {
        const asMFN = startSteppedToCurrent.monthAsMFN
        if (mfnTo.isDatedPlan) {
          const calendarMonth = mfnTo.calendarMonth(asMFN)
          return asMFN === 0
            ? {
                type: 'now',
                monthOfEntry: { isDatedPlan: true, calendarMonth },
              }
            : { type: 'calendarMonth', calendarMonth }
        } else {
          assert(startMonth.value.type === 'numericAge')
          return fGet(mfnTo.numericAge[startMonth.value.person])(asMFN)
        }
      }),
    },
    baseAmount:startSteppedToCurrent.value,
  }
}

const _stepRecurringAmountToCurrent = (
  curr: { monthAsMFN: number; value: number },
  everyXMonths: _Orig['everyXMonths'],
  delta: _Orig['delta'],
): { monthAsMFN: number; value: number } => {
  const result = _.cloneDeep(curr)
  while (result.monthAsMFN < 0) {
    result.monthAsMFN += everyXMonths
    result.value +=
      delta === null
        ? 0
        : (delta.by.type === 'percent'
            ? curr.value * delta.by.percent
            : noCase(delta.by.type)) *
          (delta.every === 'recurrence'
            ? 1
            : delta.every === 'year'
              ? // TODO: Unimplemented. Consider how to convert percent per year
                // to percent per recurrence
                assertFalse()
              : noCase(delta.every))
  }
  return {
    monthAsMFN: result.monthAsMFN,
    value: _.round(result.value),
  }
}

export const _forTest = {
  _getNormalizedDuration,
  _squashValidRangeAsMFNGivenStart,
  _squashValidRangeAsMFNGivenEnd,
  _stepStartToCurrent,
  _stepRecurringAmountToCurrent,
}
