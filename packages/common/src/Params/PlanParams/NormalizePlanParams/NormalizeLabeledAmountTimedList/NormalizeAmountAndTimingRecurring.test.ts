import _, { now } from 'lodash'
import { getMonthToMFN, normalizeAges } from '../NormalizeAges'
import {
  NormalizedAmountAndTimingRecurring,
  _forTest,
  normalizeAmountAndTimingRecurring,
} from './NormalizeAmountAndTimingRecurring'
import { normalizedMonthErrorMsg } from './NormalizedMonth'
import { assertFalse, fGet } from '../../../../Utils'
import { Month } from '../../PlanParams'
import { CalendarMonthFns } from '../../../../Misc/CalendarMonthFns'
const {
  _getNormalizedDuration,
  _squashValidRangeAsMFNGivenStart,
  _squashValidRangeAsMFNGivenEnd,
  _stepStartToCurrent,
  _stepRecurringAmountToCurrent,
} = _forTest
describe('NormalizeAmountAndTimingRecurring', () => {
  describe('_getNormalizedDuration', () => {
    describe('startAndDuration', () => {
      test('no error', () => {
        expect(
          _getNormalizedDuration(
            'startAndDuration',
            6,
            { inMonths: 6 },
            { includingLocalConstraints: { start: 0, end: 10 } },
          ),
        ).toEqual({
          type: 'normalizedDuration',
          isInThePast: false,
          asMFN: 6,
          baseValue: { inMonths: 6 },
          validRangeInMonths: {
            includingLocalConstraints: { start: 0, end: 10 },
          },
          errorMsg: null,
        })
      })

      test('out of range', () => {
        expect(
          _getNormalizedDuration(
            'startAndDuration',
            6,
            { inMonths: 6 },
            { includingLocalConstraints: { start: 0, end: 3 } },
          ),
        ).toEqual({
          type: 'normalizedDuration',
          isInThePast: false,
          asMFN: 6,
          baseValue: { inMonths: 6 },
          validRangeInMonths: {
            includingLocalConstraints: { start: 0, end: 3 },
          },
          errorMsg: normalizedMonthErrorMsg.outOfRange,
        })
      })
    })
    describe('endAndDuration', () => {
      test('no error', () => {
        expect(
          _getNormalizedDuration(
            'endAndDuration',
            6,
            { inMonths: 6 },
            { includingLocalConstraints: { start: 0, end: 10 } },
          ),
        ).toEqual({
          type: 'normalizedDuration',
          isInThePast: false,
          asMFN: 6,
          baseValue: { inMonths: 6 },
          validRangeInMonths: {
            includingLocalConstraints: { start: 2, end: 12 },
          },
          errorMsg: null,
        })
      })
      test('out of range', () => {
        expect(
          _getNormalizedDuration(
            'endAndDuration',
            6,
            { inMonths: 6 },
            { includingLocalConstraints: { start: 0, end: 5 } },
          ),
        ).toEqual({
          type: 'normalizedDuration',
          isInThePast: false,
          asMFN: 6,
          baseValue: { inMonths: 6 },
          validRangeInMonths: {
            includingLocalConstraints: { start: 7, end: 12 },
          },
          errorMsg: normalizedMonthErrorMsg.outOfRange,
        })
      })
    })
  })

  test.each([
    [7, { start: 5, end: 6 }, null],
    [6, { start: 5, end: 6 }, { start: 6, end: 6 }],
    [5, { start: 5, end: 6 }, { start: 5, end: 6 }],
    [4, { start: 5, end: 6 }, { start: 5, end: 6 }],
  ])('_squashValidRangeAsMFNGivenStart', (start, range, result) => {
    expect(_squashValidRangeAsMFNGivenStart({ asMFN: start }, range)).toEqual(
      result,
    )
  })

  test.each([
    [7, { start: 5, end: 6 }, { start: 5, end: 6 }],
    [6, { start: 5, end: 6 }, { start: 5, end: 6 }],
    [5, { start: 5, end: 6 }, { start: 5, end: 5 }],
    [4, { start: 5, end: 6 }, null],
  ])('_squashValidRangeAsMFNGivenEnd', (end, range, result) => {
    expect(_squashValidRangeAsMFNGivenEnd({ asMFN: end }, range)).toEqual(
      result,
    )
  })

  type oneTo12 = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12
  describe('_stepRecurringAmountToCurrent', () => {
    test.each([
      [-1, 1, 1 as oneTo12, 0, 1],
      [-1, 1, 2 as oneTo12, 1, 1],
    ])(
      'no delta',
      (monthAsMFN, value, everyXMonths, resultMonthAsMFN, resultValue) => {
        expect(
          _stepRecurringAmountToCurrent(
            { monthAsMFN, value },
            everyXMonths,
            null,
          ),
        ).toEqual({ monthAsMFN: resultMonthAsMFN, value: resultValue })
      },
    )

    test.each([
      [-1, 101, 1 as oneTo12, 0, _.round(101 * 1.1)],
      [-1, 101, 2 as oneTo12, 1, _.round(101 * 1.1)],
    ])(
      'delta:by .1 percent every recurrance',
      (monthAsMFN, value, everyXMonths, resultMonthAsMFN, resultValue) => {
        expect(
          _stepRecurringAmountToCurrent({ monthAsMFN, value }, everyXMonths, {
            by: { type: 'percent', percent: 0.1 },
            every: 'recurrence',
          }),
        ).toEqual({ monthAsMFN: resultMonthAsMFN, value: resultValue })
      },
    )
  })

  describe('_stepStartToCurrent', () => {
    test('not in past', () => {
      expect(
        _stepStartToCurrent(
          {
            asMFNPastNotElided: 3,
            value: {
              type: 'calendarMonth',
              calendarMonth: { year: 0, month: 0 },
            },
          },
          1,
          101,
          1,
          { by: { type: 'percent', percent: 0.1 }, every: 'recurrence' },
          { isDatedPlan: true, calendarMonth: () => assertFalse() },
        ),
      ).toEqual({
        month: {
          asMFN: 3,
          value: {
            type: 'calendarMonth',
            calendarMonth: { year: 0, month: 0 },
          },
        },
        baseAmount: 101,
      })
    })
    const nowAsCalendarDay = { year: 2024, month: 3 }
    const mfnToCalendarMonth = CalendarMonthFns.getFromMFN(nowAsCalendarDay)

    test.each([
      [1, 1 as oneTo12, [0, 101 * 1.1]],
      [1, 2 as oneTo12, [1, 101 * 1.1]],
      [1, 3 as oneTo12, null],
    ])('in past %#', (end, recurrance, result) => {
      expect(
        _stepStartToCurrent(
          {
            asMFNPastNotElided: -1,
            value: {
              type: 'calendarMonth',
              calendarMonth: mfnToCalendarMonth(-3),
            },
          },
          end,
          101,
          recurrance,
          { by: { type: 'percent', percent: 0.1 }, every: 'recurrence' },
          { isDatedPlan: true, calendarMonth: mfnToCalendarMonth },
        ),
      ).toEqual(
        result
          ? {
              month: {
                asMFN: result[0],
                value:
                  result[0] === 0
                    ? {
                        type: 'now',
                        monthOfEntry: {
                          isDatedPlan: true,
                          calendarMonth: mfnToCalendarMonth(0),
                        },
                      }
                    : {
                        type: 'calendarMonth',
                        calendarMonth: mfnToCalendarMonth(fGet(result[0])),
                      },
              },
              baseAmount: _.round(fGet(result[1])),
            }
          : null,
      )
    })
  })

  describe('normalizeAmountAndTimingRecurring', () => {
    const nowAsCalendarDay = { year: 2024, month: 3 }
    const mfnToCalendarMonth = CalendarMonthFns.getFromMFN(nowAsCalendarDay)
    const mfnToMonth = (mfn: number): Month => ({
      type: 'calendarMonth',
      calendarMonth: mfnToCalendarMonth(mfn),
    })
    const ages = normalizeAges(
      {
        withPartner: false,
        person1: {
          ages: {
            type: 'retiredWithNoRetirementDateSpecified' as const,
            currentAgeInfo: {
              isDatedPlan: true,
              monthOfBirth: mfnToCalendarMonth(-10),
            },
            maxAge: { inMonths: 20 },
          },
        },
      },
      nowAsCalendarDay,
    )
    const monthToMFN = getMonthToMFN(nowAsCalendarDay, ages)
    const validRangeAsMFN = { start: 1, end: 5 }

    test.each([
      [3, 2, [2, 2, { start: 1, end: 3 }, null, null]],
      [
        6,
        2,
        [5, 2, { start: 2, end: 6 }, null, normalizedMonthErrorMsg.outOfRange],
      ],
      [
        3,
        5,
        [0, 4, { start: 1, end: 3 }, normalizedMonthErrorMsg.outOfRange, null],
      ],
    ])('endAndDuration', (endAsMFN, duration, result) => {
      const r = normalizeAmountAndTimingRecurring(
        {
          type: 'recurring',
          monthRange: {
            type: 'endAndDuration',
            end: mfnToMonth(endAsMFN),
            duration: { inMonths: duration },
          },
          everyXMonths: 3,
          delta: null,
          baseAmount: 100,
        },
        validRangeAsMFN,
        monthToMFN,
        ages,
        nowAsCalendarDay,
      )

      if (!result) {
        expect(r).toBeNull()
      } else {
        const [
          resultStartAsMFN,
          resultDuration,
          resultStartRange,
          resultStartError,
          resultEndError,
        ] = result
        expect(r).toEqual({
          type: 'recurring',
          monthRange: {
            type: 'endAndDuration',
            end: {
              type: 'normalizedMonth',
              isInThePast: false,
              asMFN: endAsMFN,
              baseValue: mfnToMonth(endAsMFN),
              validRangeAsMFN: {
                includingLocalConstraints: validRangeAsMFN,
                excludingLocalConstraints: validRangeAsMFN,
              },
              errorMsg: resultEndError,
            },
            duration: {
              type: 'normalizedDuration',
              isInThePast: false,
              asMFN: resultStartAsMFN,
              baseValue: { inMonths: resultDuration },
              validRangeInMonths: {
                includingLocalConstraints: resultStartRange,
              },
              errorMsg: resultStartError,
            },
            validRangeAsMFN,
          },
          everyXMonths: 3,
          baseAmount: 100,
          delta: null,
        } as NormalizedAmountAndTimingRecurring)
      }
    })

    test.each([
      [1, 1, [1, 1, { start: 1, end: 5 }, null, null] as const],
      [
        6,
        1,
        [
          6,
          1,
          null,
          normalizedMonthErrorMsg.outOfRange,
          normalizedMonthErrorMsg.outOfRange,
        ] as const,
      ],
      [-1, 4, [2, 1, { start: 1, end: 4 }, null, null] as const],
      [
        -1,
        14,
        [
          2,
          11,
          { start: 1, end: 4 },
          null,
          normalizedMonthErrorMsg.outOfRange,
        ] as const,
      ],
      [-1, 3, null],
    ])('startAndDuration', (startAsMFN, duration, result) => {
      const r = normalizeAmountAndTimingRecurring(
        {
          type: 'recurring',
          monthRange: {
            type: 'startAndDuration',
            start: mfnToMonth(startAsMFN),
            duration: { inMonths: duration },
          },
          everyXMonths: 3,
          delta: null,
          baseAmount: 100,
        },
        validRangeAsMFN,
        monthToMFN,
        ages,
        nowAsCalendarDay,
      )

      if (!result) {
        expect(r).toBeNull()
      } else {
        const [
          resultStartAsMFN,
          resultEndDuration,
          resultEndRange,
          resultStartError,
          resultEndError,
        ] = result
        expect(r).toEqual({
          type: 'recurring',
          monthRange: {
            type: 'startAndDuration',
            start: {
              type: 'normalizedMonth',
              isInThePast: false,
              asMFN: resultStartAsMFN,
              baseValue: mfnToMonth(resultStartAsMFN),
              validRangeAsMFN: {
                includingLocalConstraints: validRangeAsMFN,
                excludingLocalConstraints: validRangeAsMFN,
              },
              errorMsg: resultStartError,
            },
            duration: {
              type: 'normalizedDuration',
              isInThePast: false,
              asMFN: startAsMFN + duration - 1,
              baseValue: { inMonths: resultEndDuration },
              validRangeInMonths: {
                includingLocalConstraints: resultEndRange,
              },
              errorMsg: resultEndError,
            },
            validRangeAsMFN,
          },
          everyXMonths: 3,
          baseAmount: 100,
          delta: null,
        } as NormalizedAmountAndTimingRecurring)
      }
    })

    test.each([
      [-1, 2, [2, { start: 2, end: validRangeAsMFN.end }, null, null] as const],
      [-1, -1, null],
      [1, 2, [1, { start: 1, end: validRangeAsMFN.end }, null, null] as const],
      [
        1,
        0,
        [
          1,
          { start: 1, end: validRangeAsMFN.end },
          null,
          normalizedMonthErrorMsg.outOfRange,
        ] as const,
      ],
      [
        8,
        10,
        [
          8,
          null,
          normalizedMonthErrorMsg.outOfRange,
          normalizedMonthErrorMsg.outOfRange,
        ] as const,
      ],
    ])('startAndEnd', (startAsMFN, endAsMFN, result) => {
      const r = normalizeAmountAndTimingRecurring(
        {
          type: 'recurring',
          monthRange: {
            type: 'startAndEnd',
            start: mfnToMonth(startAsMFN),
            end: mfnToMonth(endAsMFN),
          },
          everyXMonths: 3,
          delta: null,
          baseAmount: 100,
        },
        validRangeAsMFN,
        monthToMFN,
        ages,
        nowAsCalendarDay,
      )
      if (!result) {
        expect(r).toBeNull()
      } else {
        const [
          resultStartAsMFN,
          resultEndRange,
          resultStartError,
          resultEndError,
        ] = result
        expect(r).toEqual({
          type: 'recurring',
          monthRange: {
            type: 'startAndEnd',
            start: {
              type: 'normalizedMonth',
              isInThePast: false,
              asMFN: resultStartAsMFN,
              baseValue: mfnToMonth(resultStartAsMFN),
              validRangeAsMFN: {
                includingLocalConstraints: validRangeAsMFN,
                excludingLocalConstraints: validRangeAsMFN,
              },
              errorMsg: resultStartError,
            },
            end: {
              type: 'normalizedMonth',
              isInThePast: false,
              asMFN: endAsMFN,
              baseValue: mfnToMonth(endAsMFN),
              validRangeAsMFN: {
                includingLocalConstraints: resultEndRange,
                excludingLocalConstraints: validRangeAsMFN,
              },
              errorMsg: resultEndError,
            },
            validRangeAsMFN,
          },
          everyXMonths: 3,
          baseAmount: 100,
          delta: null,
        } as NormalizedAmountAndTimingRecurring)
      }
    })
  })
})
