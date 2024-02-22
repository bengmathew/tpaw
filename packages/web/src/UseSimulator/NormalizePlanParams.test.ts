import { block, getZonedTimeFns } from '@tpaw/common'
import {
  _getAgesForPerson,
  _getAsMFNForCalendarMonth,
  _getValidMonthRangeForFutureSavingsAsMFN,
  _getValidMonthRangeForIncomeDuringRetirementAsMFN,
} from './NormalizePlanParams'

describe('NormalizePlanParams', () => {
  describe('_getAsMFNForCalendarMonth', () => {
    test.each([
      [2020, 1, -12],
      [2021, 1, 0],
      [2022, 1, 12],
    ])('_getAsMFNForCalendarMonth', (year, month, result) => {
      const fn = _getAsMFNForCalendarMonth({ year: 2021, month: 1 })
      expect(fn({ year, month })).toBe(result)
    })
  })

  describe('_getAgesForPerson', () => {
    const nowCalendarMonth = { year: 2021, month: 1 }

    const asMFNForCalendarMonth = _getAsMFNForCalendarMonth(nowCalendarMonth)

    test('retiredWithNoRetirementDateSpecified', () => {
      expect(
        _getAgesForPerson(
          {
            type: 'retiredWithNoRetirementDateSpecified',
            monthOfBirth: {
              year: nowCalendarMonth.year - 10,
              month: nowCalendarMonth.month,
            },
            maxAge: { inMonths: 20 * 12 + 3 },
          },
          asMFNForCalendarMonth,
        ),
      ).toEqual({
        monthOfBirthAsMFN: -10 * 12,
        maxAgeAsMFN: 20 * 12 + 3 - 10 * 12,
        retirement: {
          ageAsMFNIfInFutureElseNull: null,
          ageAsMFNIfSpecifiedElseNull: null,
          isRetired: true,
        },
      })
    })
    test.each([
      [10 * 12 + 1, false],
      [10 * 12, true],
      [10 * 12 - 1, true],
    ])('retirementDateSpecified', (retirementAge, isRetired) => {
      expect(
        _getAgesForPerson(
          {
            type: 'retirementDateSpecified',
            monthOfBirth: {
              year: nowCalendarMonth.year - 10,
              month: nowCalendarMonth.month,
            },
            retirementAge: { inMonths: retirementAge },
            maxAge: { inMonths: 20 * 12 + 3 },
          },
          asMFNForCalendarMonth,
        ),
      ).toEqual({
        monthOfBirthAsMFN: -10 * 12,
        maxAgeAsMFN: 20 * 12 + 3 - 10 * 12,
        retirement: block(() => {
          const asMFN = retirementAge - 10 * 12
          return {
            ageAsMFNIfSpecifiedElseNull: asMFN,
            ageAsMFNIfInFutureElseNull: asMFN > 0 ? asMFN : null,
            isRetired,
          }
        }),
      })
    })
  })

  describe('_getValidMonthRangesForFutureSavingsAsMFN', () => {
    test.each([
      [null, null],
      [1, { start: 0, end: 0 }],
    ])('noPartner', (ageAsMFNIfInFutureElseNull, result) => {
      expect(
        _getValidMonthRangeForFutureSavingsAsMFN(
          { retirement: { ageAsMFNIfInFutureElseNull } },
          null,
        ),
      ).toEqual(result)
    })
    test.each([
      [null, null, null],
      [null, 1, { start: 0, end: 0 }],
      [1, null, { start: 0, end: 0 }],
      [1, 2, { start: 0, end: 1 }],
    ])(
      'withPartner',
      (
        person1AgeAsMFNIfInFutureElseNull,
        person2AgeAsMFNIfInFutureElseNull,
        result,
      ) => {
        expect(
          _getValidMonthRangeForFutureSavingsAsMFN(
            {
              retirement: {
                ageAsMFNIfInFutureElseNull: person1AgeAsMFNIfInFutureElseNull,
              },
            },
            {
              retirement: {
                ageAsMFNIfInFutureElseNull: person2AgeAsMFNIfInFutureElseNull,
              },
            },
          ),
        ).toEqual(result)
      },
    )
  })

  describe('_getValidMonthRangesForIncomeDuringRetirementAsMFN', () => {
    test.each([
      [null, 0],
      [1, 1],
    ])('noPartner', (ageAsMFNIfInFutureElseNull, start) => {
      expect(
        _getValidMonthRangeForIncomeDuringRetirementAsMFN(
          { retirement: { ageAsMFNIfInFutureElseNull } },
          null,
          { lastMonthAsMFN: 10 },
        ),
      ).toEqual({ start, end: 10 })
    })

    test.each([
      [null, null, 0],
      [null, 1, 0],
      [1, null, 0],
      [1, 2, 1],
    ])(
      'withPartner',
      (
        person1AgeAsMFNIfInFutureElseNull,
        person2AgeAsMFNIfInFutureElseNull,
        start,
      ) => {
        expect(
          _getValidMonthRangeForIncomeDuringRetirementAsMFN(
            {
              retirement: {
                ageAsMFNIfInFutureElseNull: person1AgeAsMFNIfInFutureElseNull,
              },
            },
            {
              retirement: {
                ageAsMFNIfInFutureElseNull: person2AgeAsMFNIfInFutureElseNull,
              },
            },
            { lastMonthAsMFN: 10 },
          ),
        ).toEqual({ start, end: 10 })
      },
    )
  })
})


