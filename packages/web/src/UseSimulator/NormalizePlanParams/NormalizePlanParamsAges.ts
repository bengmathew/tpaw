import {
  CalendarMonth,
  CalendarMonthFns,
  InMonths,
  Month,
  PLAN_PARAMS_CONSTANTS,
  Person,
  PlanParams,
  assert,
  block,
  fGet,
  noCase,
} from '@tpaw/common'
import { SimpleRange } from '../../Utils/SimpleRange'

export type ToMFN = ReturnType<typeof getToMFN>

export type NormalizedAges = ReturnType<typeof normalizePlanParamsAges>
export const normalizePlanParamsAges = (
  orig: PlanParams['people'],
  nowAsCalendarMonth: CalendarMonth,
) => {
  const calendarMonthToMFN = CalendarMonthFns.getToMFN(nowAsCalendarMonth)
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

export const getToMFN = ({
  nowAs,
  ages: { person1, person2 },
}: {
  nowAs: { calendarMonth: CalendarMonth }
  ages: {
    person1: ReturnType<typeof _forPerson>
    person2: ReturnType<typeof _forPerson> | null
  }
}) => {
  const calendarMonthToMFN = CalendarMonthFns.getToMFN(nowAs.calendarMonth)
  return {
    forCalendarMonth: calendarMonthToMFN,
    forMonth: block(() => {
      const pastNotElided = (month: Month) => {
        switch (month.type) {
          case 'calendarMonthAsNow':
            return calendarMonthToMFN(month.monthOfEntry)
          case 'calendarMonth':
            return calendarMonthToMFN(month.calendarMonth)
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
            return person.monthOfBirth.asMFN + month.age.inMonths
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
    }),
    inverse: {
      nowAsCalendarMonth: nowAs.calendarMonth,
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
}) => ({
  person1: (mfn: number) => person1.currentAge.inMonths + mfn,
  person2: person2 ? (mfn: number) => person2.currentAge.inMonths + mfn : null,
})

export type NormalizedCalendarMonth = {
  asMFN: number
  baseValue: CalendarMonth
  validRangeAsMFN: { includingLocalConstraints: SimpleRange }
}

const _forPerson = (
  orig: Person['ages'],
  asMFNForCalendarMonth: (x: CalendarMonth) => number,
) => {
  const monthOfBirth: NormalizedCalendarMonth = {
    asMFN: asMFNForCalendarMonth(orig.monthOfBirth),
    baseValue: orig.monthOfBirth,
    validRangeAsMFN: {
      includingLocalConstraints: {
        start: asMFNForCalendarMonth({
          year: PLAN_PARAMS_CONSTANTS.earliestYearOfBirth,
          month: 1,
        }),
        end: 0,
      },
    },
  }
  const maxAge = {
    asMFN: monthOfBirth.asMFN + orig.maxAge.inMonths,
    baseValue: orig.maxAge,
    // range:
  }
  const currentAge = { inMonths: -monthOfBirth.asMFN }
  const retirement = block(() => {
    switch (orig.type) {
      case 'retiredWithNoRetirementDateSpecified':
        return {
          ageIfInFuture: null as {
            asMFN: number
            baseValue: InMonths
            validRangeInMonths: SimpleRange
          } | null,
          ageAsMFNIfSpecifiedElseNull: null as number | null,
          numMonthsLeft: maxAge.asMFN,
          isRetired: true,
        }
      case 'retirementDateSpecified':
        const ageIfSpecified = {
          asMFN: monthOfBirth.asMFN + orig.retirementAge.inMonths,
          baseValue: orig.retirementAge,
        }
        const ageIfInFuture = ageIfSpecified.asMFN > 0 ? ageIfSpecified : null
        return {
          ageIfInFuture: ageIfInFuture
            ? {
                ...ageIfInFuture,
                validRangeInMonths: {
                  start: currentAge.inMonths + 1,
                  end: maxAge.baseValue.inMonths - 1,
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
    monthOfBirth,
    currentAge,
    maxAge: {
      ...maxAge,
      validRangeInMonths: {
        start: retirement.ageIfInFuture
          ? retirement.ageIfInFuture.baseValue.inMonths + 1
          : currentAge.inMonths + 2, // Leave space for retirement
        end: PLAN_PARAMS_CONSTANTS.maxAgeInMonths,
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
