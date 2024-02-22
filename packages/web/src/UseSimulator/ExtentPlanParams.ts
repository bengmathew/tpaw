import {
  CalendarMonth,
  DialogPosition,
  GlidePath,
  InMonths,
  Month,
  MonthRange,
  Person,
  PlanParams,
  calendarMonthFromTime,
  fGet,
  getZonedTimeFns,
  linearFnFomPoints,
  planParamsFns,
  assert,
  noCase,
} from '@tpaw/common'
import _ from 'lodash'
import { SimpleRange } from '../Utils/SimpleRange'
import { yourOrYourPartners } from '../Utils/YourOrYourPartners'

const {
  getDialogPositionEffective,
  getIsFutureSavingsAllowed,
  getNextDialogPosition,
} = planParamsFns

export type PlanParamsExtended = ReturnType<typeof extendPlanParams>
export const extendPlanParams = (
  planParams: PlanParams,
  currentTimestamp: number,
  ianaTimezoneName: string,
) => {
  const { monthRangeEdge } = extendPlanParams

  const getZonedTime = getZonedTimeFns(ianaTimezoneName)
  const currentDateTime = getZonedTime(currentTimestamp)

  const currentMonth = calendarMonthFromTime(currentDateTime)
  const getCurrentAge = ({ year, month }: CalendarMonth) => ({
    inMonths: fGet(
      currentDateTime
        .startOf('month')
        .diff(getZonedTime.fromObject({ year, month }), 'months')
        .toObject().months,
    ),
  })

  const calendarMonthToMonthsFromNow = ({ year, month }: CalendarMonth) =>
    fGet(
      getZonedTime
        .fromObject({ year, month })
        .startOf('month')
        .diff(currentDateTime.startOf('month'), 'months')
        .toObject().months,
    )

  const monthsFromNowToCalendarMonth = (mfn: number) =>
    calendarMonthFromTime(currentDateTime.plus({ month: mfn }))

  const months = (() => {
    const person = (person: 'person1' | 'person2') => ({
      lastWorkingMonth: {
        type: 'namedAge',
        person,
        age: 'lastWorkingMonth',
      } as Month,
      retirement: { type: 'namedAge', person, age: 'retirement' } as Month,
      max: { type: 'namedAge', person, age: 'max' } as Month,
      numericAge: (age: InMonths): Extract<Month, { type: 'numericAge' }> => ({
        type: 'numericAge',
        person,
        age,
      }),
    })

    const calendarMonth = (calendarMonth: CalendarMonth): Month => ({
      type: 'calendarMonth',
      calendarMonth,
    })
    return {
      now: {
        type: 'calendarMonthAsNow',
        monthOfEntry: currentMonth,
      } as Month,
      calendarMonth,
      person1: person('person1'),
      person2: person('person2'),
    }
  })()

  const pickPerson = (person: 'person1' | 'person2') => {
    if (person === 'person1') return planParams.people.person1
    assert(planParams.people.withPartner)
    return planParams.people.person2
  }

  const currentAges = {
    person1: getCurrentAge(planParams.people.person1.ages.monthOfBirth),
    person2: planParams.people.withPartner
      ? getCurrentAge(planParams.people.person2.ages.monthOfBirth)
      : null,
  }

  // Use precomputed values because the calculation is too slow for use in
  // animation.
  const getCurrentAgeOfPerson = (person: 'person1' | 'person2') =>
    fGet(currentAges[person])

  const asMonthsFromNow_Month = (month: Month): number => {
    if (month.type === 'calendarMonthAsNow')
      return calendarMonthToMonthsFromNow(month.monthOfEntry)
    if (month.type === 'calendarMonth')
      return calendarMonthToMonthsFromNow(month.calendarMonth)

    const person = pickPerson(month.person)
    const fromAgeInMonths = (age: InMonths) =>
      age.inMonths - getCurrentAgeOfPerson(month.person).inMonths

    switch (month.type) {
      case 'numericAge':
        return fromAgeInMonths(month.age)
      case 'namedAge':
        // Consider throwing instead of effectiveRetirementAge, because if a
        // person is retired with no retirement date specified, then no month
        // *should* be keyed to lastWorkingMonth or retirement (this is enforced
        // by the planParamsGuard).
        const effectiveRetirementAge =
          person.ages.type === 'retiredWithNoRetirementDateSpecified'
            ? getCurrentAgeOfPerson(month.person)
            : person.ages.retirementAge
        switch (month.age) {
          case 'lastWorkingMonth':
            return fromAgeInMonths({
              inMonths: effectiveRetirementAge.inMonths - 1,
            })
          case 'retirement':
            return fromAgeInMonths(effectiveRetirementAge)
          case 'max':
            return fromAgeInMonths(person.ages.maxAge)
          default:
            noCase(month)
        }
      default:
        noCase(month)
    }
  }

  const asMonthsFromNow_MonthRange = (
    monthRange: MonthRange,
  ): { start: number; end: number } => {
    const start =
      monthRange.type !== 'endAndNumMonths'
        ? asMFN(monthRange.start)
        : asMFN(monthRange.end) - (monthRange.numMonths - 1)
    const end =
      monthRange.type !== 'startAndNumMonths'
        ? asMFN(monthRange.end)
        : asMFN(monthRange.start) + (monthRange.numMonths - 1)
    return { start, end }
  }
  function asMonthsFromNow(month: CalendarMonth): number
  function asMonthsFromNow(month: Month): number
  function asMonthsFromNow(month: MonthRange | { start: Month; end: Month }): {
    start: number
    end: number
  }
  function asMonthsFromNow(
    x: CalendarMonth | Month | MonthRange | { start: Month; end: Month },
  ): number | { start: number; end: number } {
    if (!('type' in x))
      return 'start' in x
        ? asMonthsFromNow_MonthRange({ type: 'startAndEnd', ...x })
        : calendarMonthToMonthsFromNow(x)
    switch (x.type) {
      case 'namedAge':
      case 'numericAge':
      case 'calendarMonth':
      case 'calendarMonthAsNow':
        return asMonthsFromNow_Month(x)
      default:
        return asMonthsFromNow_MonthRange(x)
    }
  }
  const asMFN = asMonthsFromNow

  const monthsFromNowToNumericAge = (
    mfn: number,
    person: 'person1' | 'person2',
  ): Extract<Month, { type: 'numericAge' }> =>
    months[person].numericAge({
      inMonths: mfn + getCurrentAgeOfPerson(person).inMonths,
    })

  const isPersonRetired = (person: Person | 'person1' | 'person2') => {
    const p = typeof person === 'string' ? pickPerson(person) : person
    return !isAgesNotRetired(p.ages)
  }

  const isAgesNotRetired = (
    ages: Person['ages'],
  ): ages is Extract<Person['ages'], { type: 'retirementDateSpecified' }> => {
    return (
      ages.type === 'retirementDateSpecified' &&
      ages.retirementAge.inMonths >= getCurrentAge(ages.monthOfBirth).inMonths
    )
  }

  const isFutureSavingsAllowed = getIsFutureSavingsAllowed(
    isPersonRetired('person1'),
    planParams.people.withPartner ? isPersonRetired('person2') : undefined,
  )
  const dialogPositionEffective = getDialogPositionEffective(
    planParams.dialogPositionNominal,
    isFutureSavingsAllowed,
  )
  const nextDialogPosition: DialogPosition =
    dialogPositionEffective === 'done'
      ? 'done'
      : getDialogPositionEffective(
          getNextDialogPosition(dialogPositionEffective),
          isFutureSavingsAllowed,
        )

  const longerLivedPerson = ((): 'person1' | 'person2' => {
    if (!planParams.people.withPartner) return 'person1'
    return asMonthsFromNow(months.person2.max) >
      asMonthsFromNow(months.person1.max)
      ? 'person2'
      : 'person1'
  })()

  const maxMonth = (x: Month, y?: Month) => {
    if (!y) return x
    return asMFN(x) >= asMFN(y) ? x : y
  }
  const minMonth = (x: Month, y?: Month) => {
    if (!y) return x
    return asMFN(x) <= asMFN(y) ? x : y
  }

  const maxLastWorkingMonth = planParams.people.withPartner
    ? maxMonth(months.person1.lastWorkingMonth, months.person2.lastWorkingMonth)
    : months.person1.lastWorkingMonth

  const minRetirement = planParams.people.withPartner
    ? minMonth(months.person1.retirement, months.person2.retirement)
    : months.person1.retirement

  const maxMaxAge = planParams.people.withPartner
    ? maxMonth(months.person1.max, months.person2.max)
    : months.person1.max

  const getNumMonthsRemaining = (person: 'person1' | 'person2') =>
    asMFN(months[person].max) + 1

  const numMonths = planParams.people.withPartner
    ? Math.max(
        getNumMonthsRemaining('person1'),
        getNumMonthsRemaining('person2'),
      )
    : getNumMonthsRemaining('person1')

  const getIsMonthPastMaxAge = (
    month: Exclude<Month, { type: 'calendarMonth' | 'calendarMonthAsNow' }>,
  ) => asMFN(month) > asMFN(months[month.person].max)

  const clampMonthToNow = (month: Month) =>
    asMFN(month) < 0 ? months.now : month

  // Null of range is in the past.
  const clampMonthRangeToNow = (range: MonthRange): MonthRange | null => {
    const rangeAsMFN = asMFN(range)
    if (rangeAsMFN.end < 0 && rangeAsMFN.start < 0) return null
    switch (range.type) {
      case 'startAndEnd': {
        return {
          ...range,
          start: clampMonthToNow(range.start),
          end: clampMonthToNow(range.end),
        }
      }
      case 'startAndNumMonths': {
        const start = clampMonthToNow(range.start)
        const numMonths = rangeAsMFN.end - asMFN(start) + 1
        assert(numMonths > 0)
        return {
          type: 'startAndNumMonths',
          start,
          numMonths,
        }
      }
      case 'endAndNumMonths': {
        const numMonths = Math.min(range.numMonths, rangeAsMFN.end + 1)
        assert(numMonths > 0)
        return {
          type: 'endAndNumMonths',
          end: range.end,
          numMonths,
        }
      }
    }
  }

  const monthRangeBoundsCheck = (x: MonthRange, bounds: SimpleRange) => {
    const inRange = (edge: 'start' | 'end') =>
      _.inRange(asMFN(x)[edge], bounds.start, bounds.end + 1)
    const start = inRange('start') ? ('ok' as const) : ('outOfBounds' as const)
    const end = inRange('end')
      ? asMFN(x).end < asMFN(x).start
        ? ('endBeforeStart' as const)
        : ('ok' as const)
      : ('outOfBounds' as const)
    const errorMsgs = [] as string[]

    switch (start) {
      case 'ok':
        break
      case 'outOfBounds':
        errorMsgs.push('Start month is out of range.')
        break
      default:
        noCase(start)
    }
    switch (end) {
      case 'ok':
        break
      case 'endBeforeStart':
        errorMsgs.push('End month is before start.')
        break
      case 'outOfBounds':
        errorMsgs.push('End month is out of range.')
        break
      default:
        noCase(end)
    }
    return { start, end, errorMsgs }
  }

  const glidePathIntermediateValidated = (
    intermediateIn: GlidePath['intermediate'],
  ) => {
    const withoutIssues = _.sortBy(
      _.values(intermediateIn)
        .sort((a, b) => a.indexToSortByAdded - b.indexToSortByAdded)
        .map((x) => ({ ...x, monthAsMFN: asMFN(x.month) })),
      (x) => x.monthAsMFN,
    )

    const intermediate = withoutIssues.map((x, i) => ({
      month: x.month,
      stocks: x.stocks,
      indexToSortByAdded: x.indexToSortByAdded,
      id: x.id,
      issue:
        x.monthAsMFN < 0
          ? ('before' as const)
          : x.monthAsMFN > numMonths - 1
            ? ('after' as const)
            : x.monthAsMFN === numMonths - 1 ||
                x.monthAsMFN === 0 ||
                (i !== 0 && withoutIssues[i - 1].monthAsMFN === x.monthAsMFN)
              ? ('duplicate' as const)
              : ('none' as const),
    }))

    return intermediate
  }

  const validMonthRangeAsMFN = (
    type: 'future-savings' | 'income-during-retirement' | 'extra-spending',
  ) =>
    type === 'future-savings'
      ? asMFN({ start: months.now, end: maxLastWorkingMonth })
      : type === 'income-during-retirement'
        ? asMFN({ start: minRetirement, end: maxMaxAge })
        : type === 'extra-spending'
          ? asMFN({ start: months.now, end: maxMaxAge })
          : noCase(type)

  const monthRangeLength = (monthRange: MonthRange) =>
    monthRange.type === 'startAndEnd'
      ? asMFN(monthRange.end) - asMFN(monthRange.start) + 1
      : monthRange.numMonths

  
  // const monthRangeClamp = (
  //   bounds: { start: Month; end: Month },
  //   range: MonthRange,
  // ): MonthRange => {
  //   switch (range.type) {
  //     case 'startAndEnd': {
  //       return {
  //         type: 'startAndEnd',
  //         start: maxMonth(range.start, bounds.start),
  //         end: minMonth(range.end, bounds.end),
  //       }
  //     }
  //     case 'startAndNumMonths': {
  //       const start = maxMonth(range.start, bounds.start)
  //       const endAsMFN = asMFN(range).end

  //       return {
  //         type: 'startAndNumMonths',
  //         start,
  //         numMonths: Math.min(
  //           asMFN(bounds.end) + 1 - asMFN(start),
  //           range.numMonths,
  //         ),
  //       }
  //     }
  //     case 'endAndNumMonths': {
  //       const end = minMonth(range.end, bounds.end)
  //       return {
  //         type: 'endAndNumMonths',
  //         end,
  //         numMonths: Math.min(
  //           asMFN(end) + 1 - asMFN(bounds.start),
  //           range.numMonths,
  //         ),
  //       }
  //     }
  //     default:
  //       noCase(range)
  //   }
  // }

  const withdrawalStartMonth = (() => {
    const person = planParams.people.withPartner
      ? planParams.people.withdrawalStart
      : 'person1'
    return isPersonRetired(person) ? months.now : months[person].retirement
  })()

  const numRetirementMonths = numMonths - asMFN(withdrawalStartMonth)
  const withdrawalsStarted = asMFN(withdrawalStartMonth) === 0

  const getRiskToleranceFromMFN = (mfn: number) =>
    Math.max(
      0,
      pickPerson(longerLivedPerson).ages.maxAge.inMonths > 20 * 12
        ? linearFnFomPoints(
            20 * 12,
            planParams.risk.tpaw.riskTolerance.at20,
            pickPerson(longerLivedPerson).ages.maxAge.inMonths,
            planParams.risk.tpaw.riskTolerance.at20 +
              planParams.risk.tpaw.riskTolerance.deltaAtMaxAge,
          )(mfn + getCurrentAgeOfPerson(longerLivedPerson).inMonths)
        : planParams.risk.tpaw.riskTolerance.at20,
    )

  return {
    getRiskToleranceFromMFN,
    getZonedTime,
    monthsFromNowToNumericAge,
    currentTimestamp,
    ianaTimezoneName,
    isPersonRetired,
    isAgesNotRetired,
    getCurrentAgeOfPerson,
    monthsFromNowToCalendarMonth,
    calendarMonthToMonthsFromNow,
    currentMonth,
    numMonths,
    pickPerson,
    asMonthsFromNow,
    asMFN,
    minMonth,
    maxMonth,
    maxLastWorkingMonth,
    maxMaxAge,
    validMonthRangeAsMFN,
    getIsMonthPastMaxAge,
    clampMonthToNow,
    clampMonthRangeToNow,
    glidePathIntermediateValidated,
    months,
    monthRangeLength,
    monthRangeEdge,
    monthRangeBoundsCheck,
    withdrawalStartMonth,
    withdrawalsStarted,
    numRetirementMonths,
    yourOrYourPartners,
    isFutureSavingsAllowed,
    dialogPositionEffective,
    nextDialogPosition,
    planParams,
    longerLivedPerson,
  }
}

extendPlanParams.monthRangeEdge = (
  monthRange: MonthRange,
  edge: 'start' | 'end',
) =>
  'start' in monthRange && edge === 'start'
    ? monthRange.start
    : 'end' in monthRange && edge === 'end'
      ? monthRange.end
      : null
