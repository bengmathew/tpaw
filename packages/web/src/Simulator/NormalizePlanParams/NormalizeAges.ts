import {
  CalendarMonth,
  CalendarMonthFns,
  InMonths,
  Month,
  PLAN_PARAMS_CONSTANTS,
  Person,
  PickType,
  PlanParams,
  assert,
  block,
  fGet,
  noCase,
} from '@tpaw/common'
import { SimpleRange } from '../../Utils/SimpleRange'

export type MonthToMFN = ReturnType<typeof getMonthToMFN>

export type NormalizedAges = ReturnType<typeof normalizeAges>
export const normalizeAges = (
  orig: PlanParams['people'],
  nowAsCalendarDay: CalendarMonth | null,
) => {
  const calendarMonthToMFN = nowAsCalendarDay
    ? CalendarMonthFns.getToMFN(nowAsCalendarDay)
    : null
  const person1 = _forPerson(orig.person1.ages, calendarMonthToMFN)
  const person2 = orig.withPartner
    ? _forPerson(orig.person2.ages, calendarMonthToMFN)
    : null
  const simulationMonths = _getSimulationMonths(
    person1,
    person2,
    orig.withPartner ? orig.withdrawalStart : 'person1',
  )
  const validMonthRangesAsMFN = block(() => {
    const futureSavings = _getValidMonthRangeForFutureSavingsAsMFN(
      person1,
      person2,
    )

    return {
      futureSavings,
      incomeDuringRetirement: _getValidMonthRangeForIncomeDuringRetirementAsMFN(
        person1,
        person2,
        simulationMonths,
      ),
      extraSpending: _getValidMonthRangeForExtraSpendingAsMFN(simulationMonths),
    }
  })
  return {
    person1,
    person2,
    longerLivedPersonType:
      person2 && person2.maxAge.asMFN > person1.maxAge.asMFN
        ? ('person2' as const)
        : ('person1' as const),
    simulationMonths,
    validMonthRangesAsMFN,
  }
}

export const getMonthToMFN = (
  nowAsCalendarDay: CalendarMonth | null,
  {
    person1,
    person2,
  }: {
    person1: ReturnType<typeof _forPerson>
    person2: ReturnType<typeof _forPerson> | null
  },
) => {
  const calendarMonthToMFN = nowAsCalendarDay
    ? CalendarMonthFns.getToMFN(nowAsCalendarDay)
    : null

  const pastNotElided = (month: Month) => {
    switch (month.type) {
      case 'now':
        return month.monthOfEntry.isDatedPlan
          ? fGet(calendarMonthToMFN)(month.monthOfEntry.calendarMonth)
          : 0
      case 'calendarMonth':
        return fGet(calendarMonthToMFN)(month.calendarMonth)
      case 'namedAge': {
        const person = fGet({ person1, person2 }[month.person])
        switch (month.age) {
          case 'lastWorkingMonth':
            return fGet(person.retirement.ageAsMFNIfSpecifiedElseNull) - 1
          case 'retirement':
            return fGet(person.retirement.ageAsMFNIfSpecifiedElseNull)
          case 'max':
            return person.maxAge.asMFN
          default:
            noCase(month)
        }
      }
      case 'numericAge': {
        const person = fGet({ person1, person2 }[month.person])
        return person.currentAgeInfo.asMFN + month.age.inMonths
      }
      default:
        noCase(month)
    }
  }
  return {
    // May be negative.
    pastNotElided,
    // Not negative or 'inThePast'.
    pastElided: (month: Month) => {
      const asMFN = pastNotElided(month)
      return asMFN < 0 ? ('inThePast' as const) : asMFN
    },
    // Not negative.
    fNotInPast: (month: Month) => {
      const asMFN = pastNotElided(month)
      assert(asMFN >= 0)
      return asMFN
    },
  }
}

export const getFromMFNToNumericAge = ({
  ages: { person1, person2 },
}: {
  ages: {
    person1: ReturnType<typeof _forPerson>
    person2: ReturnType<typeof _forPerson> | null
  }
}) => {
  const toPerson1 = (mfn: number): PickType<Month, 'numericAge'> => ({
    type: 'numericAge',
    person: 'person1',
    age: { inMonths: person1.currentAgeInfo.inMonths + mfn },
  })
  const toPerson2 = person2
    ? (mfn: number): PickType<Month, 'numericAge'> => ({
        type: 'numericAge',
        person: 'person2',
        age: { inMonths: person2.currentAgeInfo.inMonths + mfn },
      })
    : null

  return {
    person1: toPerson1,
    person2: toPerson2,
    auto: (mfn: number) => {
      const asPerson1 = toPerson1(mfn)
      if (asPerson1.age.inMonths <= person1.maxAge.asMFN || !toPerson2)
        return asPerson1
      assert(person2)
      const asPerson2 = toPerson2(mfn)
      return asPerson2.age.inMonths <= person2.maxAge.asMFN
        ? asPerson2
        : asPerson1
    },
  }
}

export type NormalizedCalendarMonth = {
  asMFN: number
  baseValue: CalendarMonth
  validRangeAsMFN: { includingLocalConstraints: SimpleRange }
}

const _forPerson = (
  orig: Person['ages'],
  asMFNForCalendarMonth: null | ((x: CalendarMonth) => number),
) => {
  const currentAgeInfo = orig.currentAgeInfo.isDatedPlan
    ? block(() => {
        assert(orig.currentAgeInfo.isDatedPlan)

        const asMFN = fGet(asMFNForCalendarMonth)(
          orig.currentAgeInfo.monthOfBirth,
        )
        return {
          isDatedPlan: true as const,
          inMonths: -asMFN,
          asMFN,
          baseValue: orig.currentAgeInfo.monthOfBirth,
          validRangeAsMFN: {
            includingLocalConstraints: {
              start: fGet(asMFNForCalendarMonth)({
                year: PLAN_PARAMS_CONSTANTS.people.ages.person.currentAgeInfo
                  .datedPlan.earliestYearOfBirth,
                month: 1,
              }),
              end: 0,
            },
          },
        }
      })
    : {
        isDatedPlan: false as const,
        inMonths: orig.currentAgeInfo.currentAge.inMonths,
        asMFN: -orig.currentAgeInfo.currentAge.inMonths,
        baseValue: orig.currentAgeInfo.currentAge,
      }

  const maxAge = {
    asMFN: currentAgeInfo.asMFN + orig.maxAge.inMonths,
    baseValue: orig.maxAge,
    // range:
  }
  // if (maxAge.asMFN < 0) {
  //   return { isInThePast: true } as const
  // }
  const retirement = block(() => {
    switch (orig.type) {
      case 'retiredWithNoRetirementDateSpecified':
        return {
          ageIfInFuture: null as {
            asMFN: number
            baseValue: InMonths
            validRangeInMonths: { includingLocalConstraints: SimpleRange }
          } | null,
          ageAsMFNIfSpecifiedElseNull: null as number | null,
          numMonthsLeft: maxAge.asMFN,
          isRetired: true,
        }
      case 'retirementDateSpecified':
        const ageIfSpecified = {
          asMFN: currentAgeInfo.asMFN + orig.retirementAge.inMonths,
          baseValue: orig.retirementAge,
        }
        const ageIfInFuture = ageIfSpecified.asMFN > 0 ? ageIfSpecified : null
        return {
          ageIfInFuture: ageIfInFuture
            ? {
                ...ageIfInFuture,
                validRangeInMonths: {
                  includingLocalConstraints: {
                    start: currentAgeInfo.inMonths + 1,
                    end: maxAge.baseValue.inMonths - 1,
                  },
                },
              }
            : null,
          ageAsMFNIfSpecifiedElseNull: ageIfSpecified.asMFN,
          numMonthsLeft: maxAge.asMFN - (ageIfInFuture?.asMFN ?? 0),
          isRetired: ageIfInFuture === null,
        }
      default:
        noCase(orig)
    }
  })
  return {
    // isInThePast: false as const,
    currentAgeInfo: currentAgeInfo.isDatedPlan
      ? currentAgeInfo
      : {
          ...currentAgeInfo,
          validRangeInMonths: {
            includingLocalConstraints: {
              start: 0,
              end: retirement.ageIfInFuture
                ? retirement.ageIfInFuture.baseValue.inMonths - 1
                : maxAge.baseValue.inMonths - 2, // Leave space for retirement
            },
          },
        },
    maxAge: {
      ...maxAge,
      validRangeInMonths: {
        includingLocalConstraints: {
          start: retirement.ageIfInFuture
            ? retirement.ageIfInFuture.baseValue.inMonths + 1
            : currentAgeInfo.inMonths + 2, // Leave space for retirement
          end: PLAN_PARAMS_CONSTANTS.people.ages.person.maxAge,
        },
      },
    },
    retirement,
  }
}

export const _getSimulationMonths = (
  person1: ReturnType<typeof _forPerson>,
  person2: ReturnType<typeof _forPerson> | null,
  withdrawalStartsAtRetirementOf: 'person1' | 'person2',
) => {
  const lastMonthAsMFN = Math.max(
    person1.maxAge.asMFN,
    person2?.maxAge.asMFN ?? 0,
  )

  const withdrawalStartMonth = block(() => {
    const withdrawalStartPerson = fGet(
      { person1, person2 }[withdrawalStartsAtRetirementOf],
    )
    const asMFN = withdrawalStartPerson.retirement.ageIfInFuture?.asMFN ?? 0
    return { asMFN, atRetirementOf: withdrawalStartsAtRetirementOf }
  })
  const numMonths = lastMonthAsMFN + 1
  return {
    numMonths: numMonths,
    numWithdrawalMonths: numMonths - withdrawalStartMonth.asMFN,
    lastMonthAsMFN,
    withdrawalStartMonth,
  }
}

export const _getValidMonthRangeForFutureSavingsAsMFN = (
  person1: { retirement: { ageIfInFuture: { asMFN: number } | null } },
  person2: { retirement: { ageIfInFuture: { asMFN: number } | null } } | null,
) => {
  if (person2 === null) {
    return person1.retirement.ageIfInFuture === null
      ? null
      : { start: 0, end: person1.retirement.ageIfInFuture.asMFN - 1 }
  } else {
    if (
      person1.retirement.ageIfInFuture === null &&
      person2.retirement.ageIfInFuture === null
    )
      return null
    return {
      start: 0,
      end:
        Math.max(
          person1.retirement.ageIfInFuture?.asMFN ?? 0,
          person2.retirement.ageIfInFuture?.asMFN ?? 0,
        ) - 1,
    }
  }
}

const _getValidMonthRangeForIncomeDuringRetirementAsMFN = (
  person1: { retirement: { ageIfInFuture: { asMFN: number } | null } },
  person2: { retirement: { ageIfInFuture: { asMFN: number } | null } } | null,
  simulationMonths: { lastMonthAsMFN: number },
) =>
  person2 === null
    ? {
        start: person1.retirement.ageIfInFuture?.asMFN ?? 0,
        end: simulationMonths.lastMonthAsMFN,
      }
    : {
        start: Math.min(
          person1.retirement.ageIfInFuture?.asMFN ?? 0,
          person2.retirement.ageIfInFuture?.asMFN ?? 0,
        ),
        end: simulationMonths.lastMonthAsMFN,
      }

const _getValidMonthRangeForExtraSpendingAsMFN = (simulationMonths: {
  lastMonthAsMFN: number
}) => ({ start: 0, end: simulationMonths.lastMonthAsMFN })

export const _forTesting = {
  _forPerson,
  _getSimulationMonths,
  _getValidMonthRangeForFutureSavingsAsMFN,
  _getValidMonthRangeForIncomeDuringRetirementAsMFN,
  _getValidMonthRangeForExtraSpendingAsMFN,
}
