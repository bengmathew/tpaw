import { GlidePath, Month, MonthRange, Person, PlanParams } from '@tpaw/common'
import _ from 'lodash'
import { SimpleRange } from '../Utils/SimpleRange'
import { assert, noCase } from '../Utils/Utils'
import { yourOrYourPartners } from '../Utils/YourOrYourPartners'

export type PlanParamsExt = ReturnType<typeof extendPlanParams>
export const extendPlanParams = (params: PlanParams) => {
  const { months, monthRangeEdge } = extendPlanParams

  const pickPerson = (person: 'person1' | 'person2') => {
    if (person === 'person1') return params.people.person1
    assert(params.people.withPartner)
    return params.people.person2
  }

  const asMonthsFromNow_Month = (month: Month): number => {
    if (month.type === 'now') return 0
    const person = pickPerson(month.person)
    const fromAgeInMonths = (ageInMonths: number) =>
      ageInMonths - person.ages.currentMonth

    switch (month.type) {
      case 'numericAge':
        return fromAgeInMonths(month.ageInMonths)
      case 'namedAge':
        const effectiveRetirementMonth =
          person.ages.type === 'retired'
            ? person.ages.currentMonth
            : person.ages.retirementMonth
        switch (month.age) {
          case 'lastWorkingMonth':
            return fromAgeInMonths(effectiveRetirementMonth - 1)
          case 'retirement':
            return fromAgeInMonths(effectiveRetirementMonth)
          case 'max':
            return fromAgeInMonths(person.ages.maxMonth)
          default:
            noCase(month.age)
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
  function asMonthsFromNow(month: Month): number
  function asMonthsFromNow(month: MonthRange | { start: Month; end: Month }): {
    start: number
    end: number
  }
  function asMonthsFromNow(
    x: Month | MonthRange | { start: Month; end: Month },
  ): number | { start: number; end: number } {
    if (!('type' in x))
      return asMonthsFromNow_MonthRange({ type: 'startAndEnd', ...x })
    switch (x.type) {
      case 'namedAge':
      case 'numericAge':
      case 'now':
        return asMonthsFromNow_Month(x)
      default:
        return asMonthsFromNow_MonthRange(x)
    }
  }
  const asMFN = asMonthsFromNow

  const longerLivedPerson = ((): 'person1' | 'person2' => {
    const { person1 } = params.people
    if (!params.people.withPartner) return 'person1'
    const { person2 } = params.people
    const person1LifeSpan = person1.ages.maxMonth - person1.ages.currentMonth
    const person2LifeSpan = person2.ages.maxMonth - person2.ages.currentMonth
    return person2LifeSpan > person1LifeSpan ? 'person2' : 'person1'
  })()

  const maxMonth = (x: Month, y?: Month) => {
    if (!y) return x
    return asMFN(x) >= asMFN(y) ? x : y
  }
  const minMonth = (x: Month, y?: Month) => {
    if (!y) return x
    return asMFN(x) <= asMFN(y) ? x : y
  }

  const maxLastWorkingMonth = params.people.withPartner
    ? maxMonth(months.person1.lastWorkingMonth, months.person2.lastWorkingMonth)
    : months.person1.lastWorkingMonth

  const minRetirement = params.people.withPartner
    ? minMonth(months.person1.retirement, months.person2.retirement)
    : months.person1.retirement

  const maxMaxAge = params.people.withPartner
    ? maxMonth(months.person1.max, months.person2.max)
    : months.person1.max
  const numMonths = getNumMonths(params)

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
      intermediateIn.map((x) => ({ ...x, monthAsMFN: asMFN(x.month) })),
      (x) => x.monthAsMFN,
    )

    const intermediate = withoutIssues.map((x, i) => ({
      month: x.month,
      stocks: x.stocks,
      issue:
        x.monthAsMFN <= 0
          ? ('before' as const)
          : x.monthAsMFN >= numMonths - 1
          ? ('after' as const)
          : i !== 0 && withoutIssues[i - 1].monthAsMFN === x.monthAsMFN
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

  const monthRangeClamp = (
    bounds: { start: Month; end: Month },
    range: MonthRange,
  ): MonthRange => {
    switch (range.type) {
      case 'startAndEnd': {
        return {
          type: 'startAndEnd',
          start: maxMonth(range.start, bounds.start),
          end: minMonth(range.end, bounds.end),
        }
      }
      case 'startAndNumMonths': {
        const start = maxMonth(range.start, bounds.start)
        return {
          type: 'startAndNumMonths',
          start,
          numMonths: Math.min(
            asMFN(bounds.end) + 1 - asMFN(start),
            range.numMonths,
          ),
        }
      }
      case 'endAndNumMonths': {
        const end = minMonth(range.end, bounds.end)
        return {
          type: 'endAndNumMonths',
          end,
          numMonths: Math.min(
            asMFN(end) + 1 - asMFN(bounds.start),
            range.numMonths,
          ),
        }
      }
      default:
        noCase(range)
    }
  }

  const withdrawalStartMonth = (() => {
    const person = params.people.withPartner
      ? params.people.withdrawalStart
      : 'person1'
    const { ages } = pickPerson(person)
    return ages.type === 'retired' ? months.now : months[person].retirement
  })()

  const numRetirementMonths = numMonths - asMFN(withdrawalStartMonth)
  const withdrawalsStarted = asMFN(withdrawalStartMonth) === 0

  return {
    numMonths,
    pickPerson,
    asMonthsFromNow,
    asMFN,
    minMonth,
    maxMonth,
    maxLastWorkingMonth,
    minRetirement,
    maxMaxAge,
    validMonthRangeAsMFN,
    glidePathIntermediateValidated,
    months,
    monthRangeLength,
    monthRangeClamp,
    monthRangeEdge,
    monthRangeBoundsCheck,
    withdrawalStartMonth,
    withdrawalsStarted,
    numRetirementMonths,
    yourOrYourPartners,
    params,
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

extendPlanParams.months = (() => {
  const person = (person: 'person1' | 'person2') => ({
    lastWorkingMonth: {
      type: 'namedAge' as const,
      person,
      age: 'lastWorkingMonth' as const,
    },
    retirement: {
      type: 'namedAge' as const,
      person,
      age: 'retirement' as const,
    },
    max: { type: 'namedAge' as const, person, age: 'max' as const },
    numericAge: (ageInMonths: number) => ({
      type: 'numericAge' as const,
      person,
      ageInMonths,
    }),
  })
  return {
    now: { type: 'now' as const },
    person1: person('person1'),
    person2: person('person2'),
  }
})()

export const getNumMonths = (params: PlanParams) => {
  const forPerson = (person: Person) =>
    person.ages.maxMonth - person.ages.currentMonth + 1
  return Math.max(
    forPerson(params.people.person1),
    params.people.withPartner ? forPerson(params.people.person2) : 0,
  )
}

export const getWithdrawalStartAsMFN = (params: PlanParams) => {
  const person = params.people.withPartner
    ? params.people.withdrawalStart
    : 'person1'
  const { ages } = (() => {
    if (person === 'person1') return params.people.person1
    assert(params.people.withPartner)
    return params.people.person2
  })()
  return ages.type === 'retired' ? 0 : ages.retirementMonth - ages.currentMonth
}
