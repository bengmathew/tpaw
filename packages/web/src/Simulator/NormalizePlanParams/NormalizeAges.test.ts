import {
  CalendarMonthFns,
  PLAN_PARAMS_CONSTANTS,
  Person,
  block,
  fGet,
} from '@tpaw/common'
import {
  _forTesting,
  getFromMFNToNumericAge,
  normalizeAges,
} from './NormalizeAges'

const {
  _forPerson,
  _getValidMonthRangeForFutureSavingsAsMFN,
  _getValidMonthRangeForIncomeDuringRetirementAsMFN,
} = _forTesting

describe('NormalizePlanParamsAges', () => {
  describe('_forPerson', () => {
    const nowCalendarMonth = { year: 2021, month: 1 }

    const calendarMonToMFN = CalendarMonthFns.getToMFN(nowCalendarMonth)

    const validRangeAsMFNForMonthOfBirth = {
      includingLocalConstraints: {
        start: calendarMonToMFN({
          year: PLAN_PARAMS_CONSTANTS.people.ages.person.currentAgeInfo
            .datedPlan.earliestYearOfBirth,
          month: 1,
        }),
        end: 0,
      },
    }
    test('retiredWithNoRetirementDateSpecified', () => {
      const monthOfBirth = CalendarMonthFns.addMonths(
        nowCalendarMonth,
        -10 * 12,
      )
      const person: Person['ages'] = {
        type: 'retiredWithNoRetirementDateSpecified',
        currentAgeInfo: {
          isDatedPlan: true,
          monthOfBirth: monthOfBirth,
        },
        maxAge: { inMonths: 20 * 12 + 3 },
      }
      const result = _forPerson(person, calendarMonToMFN)
      expect(result).toEqual({
        currentAgeInfo: {
          isDatedPlan: true,
          inMonths: 10 * 12,
          asMFN: -10 * 12,
          baseValue: monthOfBirth,
          validRangeAsMFN: validRangeAsMFNForMonthOfBirth,
        },
        maxAge: {
          asMFN: 20 * 12 + 3 - 10 * 12,
          baseValue: person.maxAge,
          validRangeInMonths: {
            includingLocalConstraints: {
              start: 10 * 12 + 2,
              end: PLAN_PARAMS_CONSTANTS.people.ages.person.maxAge,
            },
          },
        },
        retirement: {
          ageIfInFuture: null,
          ageAsMFNIfSpecifiedElseNull: null,
          isRetired: true,
          numMonthsLeft: 20 * 12 + 3 - 10 * 12,
        },
      } as ReturnType<typeof _forPerson>)
    })

    test.each([
      [10 * 12 + 1, false],
      [10 * 12, true],
      [10 * 12 - 1, true],
    ])('retirementDateSpecified', (retirementAge, isRetired) => {
      const monthOfBirth = {
        year: nowCalendarMonth.year - 10,
        month: nowCalendarMonth.month,
      }
      const person: Person['ages'] = {
        type: 'retirementDateSpecified',
        currentAgeInfo: { isDatedPlan: true, monthOfBirth },
        retirementAge: { inMonths: retirementAge },
        maxAge: { inMonths: 20 * 12 + 3 },
      }

      expect(_forPerson(person, calendarMonToMFN)).toMatchObject({
        currentAgeInfo: {
          isDatedPlan: true,
          asMFN: -10 * 12,
          inMonths: 10 * 12,
          baseValue: monthOfBirth,
          validRangeAsMFN: validRangeAsMFNForMonthOfBirth,
        },
        maxAge: {
          asMFN: 20 * 12 + 3 - 10 * 12,
          baseValue: person.maxAge,
        },

        retirement: block(() => {
          const ageIfSpecified = {
            asMFN: retirementAge - 10 * 12,
            baseValue: person.retirementAge,
          }
          return {
            ageAsMFNIfSpecifiedElseNull: ageIfSpecified.asMFN,
            ageIfInFuture:
              ageIfSpecified.asMFN > 0
                ? {
                    ...ageIfSpecified,
                    validRangeInMonths: {
                      includingLocalConstraints: {
                        start: 10 * 12 + 1,
                        end: 20 * 12 + 3 - 1,
                      },
                    },
                  }
                : null,
            isRetired,
          }
        }),
      } as ReturnType<typeof _forPerson>)
    })
  })

  describe('_getValidMonthRangesForFutureSavingsAsMFN', () => {
    test.each([
      [null, null],
      [{ asMFN: 1 }, { start: 0, end: 0 }],
    ])('noPartner', (ageIfInFuture, result) => {
      expect(
        _getValidMonthRangeForFutureSavingsAsMFN(
          { retirement: { ageIfInFuture } },
          null,
        ),
      ).toEqual(result)
    })
    test.each([
      [null, null, null],
      [null, { asMFN: 1 }, { start: 0, end: 0 }],
      [{ asMFN: 1 }, null, { start: 0, end: 0 }],
      [{ asMFN: 1 }, { asMFN: 2 }, { start: 0, end: 1 }],
    ])('withPartner', (person1AgeIfInFuture, person2AgeIfInFuture, result) => {
      expect(
        _getValidMonthRangeForFutureSavingsAsMFN(
          {
            retirement: {
              ageIfInFuture: person1AgeIfInFuture,
            },
          },
          {
            retirement: {
              ageIfInFuture: person2AgeIfInFuture,
            },
          },
        ),
      ).toEqual(result)
    })
  })

  describe('_getValidMonthRangesForIncomeDuringRetirementAsMFN', () => {
    test.each([
      [null, 0],
      [{ asMFN: 1 }, 1],
    ])('noPartner', (ageIfInFuture, start) => {
      expect(
        _getValidMonthRangeForIncomeDuringRetirementAsMFN(
          { retirement: { ageIfInFuture } },
          null,
          { lastMonthAsMFN: 10 },
        ),
      ).toEqual({ start, end: 10 })
    })

    test.each([
      [null, null, 0],
      [null, { asMFN: 1 }, 0],
      [{ asMFN: 1 }, null, 0],
      [{ asMFN: 1 }, { asMFN: 2 }, 1],
    ])('withPartner', (person1AgeIfInFuture, person2AgeIfInFuture, start) => {
      expect(
        _getValidMonthRangeForIncomeDuringRetirementAsMFN(
          {
            retirement: {
              ageIfInFuture: person1AgeIfInFuture,
            },
          },
          {
            retirement: {
              ageIfInFuture: person2AgeIfInFuture,
            },
          },
          { lastMonthAsMFN: 10 },
        ),
      ).toEqual({ start, end: 10 })
    })
  })

  test.each([
    ['person1' as const, 0, 10],
    ['person2' as const, 0, 20],
  ])('getFromMFNToNumericAge', (personType, mfn, result) => {
    const nowAs = { calendarMonth: { year: 2024, month: 3 } }
    const mfnToCalendarMonth = CalendarMonthFns.getFromMFN(nowAs.calendarMonth)
    const ages = normalizeAges(
      {
        withPartner: true,
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
        person2: {
          ages: {
            type: 'retiredWithNoRetirementDateSpecified' as const,
            currentAgeInfo: {
              isDatedPlan: true,
              monthOfBirth: mfnToCalendarMonth(-20),
            },
            maxAge: { inMonths: 30 },
          },
        },
        withdrawalStart: 'person2',
      },
      nowAs.calendarMonth,
    )
    expect(
      fGet(getFromMFNToNumericAge({ ages })[personType])(mfn).age.inMonths,
    ).toEqual(result)
  })
})
