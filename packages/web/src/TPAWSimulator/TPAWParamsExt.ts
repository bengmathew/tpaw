import _ from 'lodash'
import {pluralize} from '../Utils/Pluralize'
import {assert, noCase} from '../Utils/Utils'
import {TPAWParams, Year, YearRange} from './TPAWParams'

export type TPAWParamsExt = ReturnType<typeof extendTPAWParams>
export const extendTPAWParams = (params: TPAWParams) => {
  const {years, yearRangeEdge} = extendTPAWParams

  const pickPerson = (person: 'person1' | 'person2') => {
    if (person === 'person1') return params.people.person1
    assert(params.people.withPartner)
    return params.people.person2
  }

  const asYearsFromNow_Year = (year: Year): number => {
    if (year.type === 'now') return 0
    const person = pickPerson(year.person)
    const {age: ageIn} = year
    const age = (() => {
      const effectiveRetirement =
        person.ages.type === 'retired'
          ? person.ages.current
          : person.ages.retirement
      if (ageIn === 'lastWorkingYear') {
        return effectiveRetirement - 1
      }
      if (ageIn === 'retirement') {
        return effectiveRetirement
      }
      if (ageIn === 'max') return person.ages.max
      return ageIn
    })()
    return age - person.ages.current
  }

  const asYearsFromNow_YearRange = (
    yearRange: YearRange
  ): {start: number; end: number} => {
    const start =
      yearRange.type !== 'endAndNumYears'
        ? asYFN(yearRange.start)
        : asYFN(yearRange.end) - (yearRange.numYears - 1)
    const end =
      yearRange.type !== 'startAndNumYears'
        ? asYFN(yearRange.end)
        : asYFN(yearRange.start) + (yearRange.numYears - 1)
    return {start, end}
  }
  function asYearsFromNow(year: Year): number
  function asYearsFromNow(year: YearRange | {start: Year; end: Year}): {
    start: number
    end: number
  }
  function asYearsFromNow(
    x: Year | YearRange | {start: Year; end: Year}
  ): number | {start: number; end: number} {
    if (!('type' in x))
      return asYearsFromNow_YearRange({type: 'startAndEnd', ...x})
    switch (x.type) {
      case 'namedAge':
      case 'numericAge':
      case 'now':
        return asYearsFromNow_Year(x)
      default:
        return asYearsFromNow_YearRange(x)
    }
  }
  const asYFN = asYearsFromNow

  const maxYear = (x: Year, y?: Year) => {
    if (!y) return x
    return asYFN(x) >= asYFN(y) ? x : y
  }
  const minYear = (x: Year, y?: Year) => {
    if (!y) return x
    return asYFN(x) <= asYFN(y) ? x : y
  }
  const yearRangeBoundsCheck = (
    x: YearRange,
    bounds: {start: Year; end: Year}
  ) => {
    const inRange = (edge: 'start' | 'end') =>
      _.inRange(asYFN(x)[edge], asYFN(bounds).start, asYFN(bounds).end + 1)
    const start = inRange('start') ? ('ok' as const) : ('outOfBounds' as const)
    const end = inRange('end')
      ? asYFN(x).end < asYFN(x).start
        ? ('endBeforeStart' as const)
        : ('ok' as const)
      : ('outOfBounds' as const)
    const errorMsgs = [] as string[]

    switch (start) {
      case 'ok':
        break
      case 'outOfBounds':
        errorMsgs.push('Start year is out of range.')
        break
      default:
        noCase(start)
    }
    switch (end) {
      case 'ok':
        break
      case 'endBeforeStart':
        errorMsgs.push('End year is before start.')
        break
      case 'outOfBounds':
        errorMsgs.push('End year is out of range.')
        break
      default:
        noCase(end)
    }
    return {start, end, errorMsgs}
  }

  const maxLastWorkingYear = params.people.withPartner
    ? maxYear(years.person1.lastWorkingYear, years.person2.lastWorkingYear)
    : years.person1.lastWorkingYear

  const minRetiremet = params.people.withPartner
    ? minYear(years.person1.retirement, years.person2.retirement)
    : years.person1.retirement

  const maxMaxAge = params.people.withPartner
    ? maxYear(years.person1.max, years.person2.max)
    : years.person1.max

  const validYearRange = (
    type: 'future-savings' | 'income-during-retirement' | 'extra-spending'
  ) =>
    type === 'future-savings'
      ? {start: years.now, end: maxLastWorkingYear}
      : type === 'income-during-retirement'
      ? {start: minRetiremet, end: maxMaxAge}
      : type === 'extra-spending'
      ? {start: years.now, end: maxMaxAge}
      : noCase(type)

  const yearRangeLength = (yearRange: YearRange) =>
    yearRange.type === 'startAndEnd'
      ? asYFN(yearRange.end) - asYFN(yearRange.start) + 1
      : yearRange.numYears

  const yearRangeClamp = (
    bounds: {start: Year; end: Year},
    range: YearRange
  ): YearRange => {
    switch (range.type) {
      case 'startAndEnd': {
        return {
          type: 'startAndEnd',
          start: maxYear(range.start, bounds.start),
          end: minYear(range.end, bounds.end),
        }
      }
      case 'startAndNumYears': {
        const start = maxYear(range.start, bounds.start)
        return {
          type: 'startAndNumYears',
          start,
          numYears: Math.min(
            asYFN(bounds.end) + 1 - asYFN(start),
            range.numYears
          ),
        }
      }
      case 'endAndNumYears': {
        const end = minYear(range.end, bounds.end)
        return {
          type: 'endAndNumYears',
          end,
          numYears: Math.min(
            asYFN(end) + 1 - asYFN(bounds.start),
            range.numYears
          ),
        }
      }
      default:
        noCase(range)
    }
  }

  const withdrawalStartYear = (() => {
    const person = params.people.withPartner
      ? params.people.withdrawalStart
      : 'person1'
    const {ages} = pickPerson(person)
    return ages.type === 'retired' ? years.now : years.person(person).retirement
  })()

  const yourOrYourPartners = (person: 'person1' | 'person2') =>
    person === 'person1' ? 'your' : "your partner's"

  const yearRangeToStr = (
    value: YearRange,
    {
      capitalize = true,
      lengthInBrackets = true,
    }: {capitalize?: boolean; lengthInBrackets?: boolean} = {}
  ) => {
    const {withPartner} = params.people

    const result: {type: 'start' | 'end' | 'other'; value: string}[] = []
    let type: 'start' | 'end' | 'other' = 'other'
    const push = (value: string) => {
      const last = _.last(result)
      if (!last || last.type !== type) {
        result.push({type, value})
      } else {
        last.value = `${last.value}${value.startsWith('.')?'':' '}${value}`
      }
    }

    const yourOrYourPartners = ({person}: {person: 'person1' | 'person2'}) =>
      push(person === 'person1' ? 'your' : "your partner's")
    const youAreOrYourPartnerIs = ({person}: {person: 'person1' | 'person2'}) =>
      push(person === 'person1' ? 'you are' : 'your partner is')

    const namedAge = (year: Extract<Year, {type: 'namedAge'}>) => {
      switch (year.age) {
        case 'lastWorkingYear':
          yourOrYourPartners(year)
          push('last working year')
          break
        case 'retirement':
          if (withPartner) yourOrYourPartners(year)
          push('retirement')
          break
        case 'max':
          yourOrYourPartners(year)
          push('max age')
          break
      }
      push(`(age ${asYFN(year) + pickPerson(year.person).ages.current})`)
    }

    const yearForDuration = (year: Year) => {
      switch (year.type) {
        case 'now':
          push('now')
          break
        case 'namedAge':
          push('at')
          namedAge(year)
          break
        case 'numericAge':
          if (withPartner) {
            push('when')
            youAreOrYourPartnerIs(year)
            push(`${year.age}`)
          } else {
            push(`at age ${year.age}`)
          }
          break
        default:
          noCase(year)
      }
    }

    switch (value.type) {
      case 'startAndEnd':
        type = 'other'
        push(capitalize ? 'From' : 'from')
        type = 'start'
        switch (value.start.type) {
          case 'now':
            push('now')
            break
          case 'namedAge':
            namedAge(value.start)
            break
          case 'numericAge':
            if (withPartner) {
              push('when')
              youAreOrYourPartnerIs(value.start)
              push(`${value.start.age}`)
            } else {
              push(`age ${value.start.age}`)
            }
            break
          default:
            noCase(value.start)
        }
        type = 'other'
        push('until')
        type = 'end'
        switch (value.end.type) {
          case 'now':
            push('now')
            break
          case 'namedAge':
            namedAge(value.end)
            break
          case 'numericAge':
            if (withPartner) {
              youAreOrYourPartnerIs(value.end)
              push(`${value.end.age}`)
            } else {
              push(`age ${value.end.age}`)
            }
            break
          default:
            noCase(value.end)
        }
        type = 'other'
        push(
          lengthInBrackets
            ? `(${pluralize(yearRangeLength(value), 'year')})`
            : `. That's ${pluralize(yearRangeLength(value), 'year')}`
        )
        break
      case 'startAndNumYears':
        type = 'other'
        push(capitalize ? 'For' : 'for')
        type = 'end'
        push(`${pluralize(value.numYears, 'year')}`)
        type = 'other'
        push('starting')
        type = 'start'
        yearForDuration(value.start)
        break
      case 'endAndNumYears':
        type = 'other'
        push(capitalize ? 'For' : 'for')
        type = 'start'
        push(`${pluralize(value.numYears, 'year')}`)
        type = 'other'
        push('ending')
        type = 'end'
        yearForDuration(value.end)
        break
      default:
        noCase(value)
    }
    type = 'other'
    push('.')
    return result
  }

  return {
    numYears: asYFN(maxMaxAge) + 1,
    pickPerson,
    asYearsFromNow,
    asYFN,
    minYear,
    maxYear,
    maxLastWorkingYear,
    minRetiremet,
    maxMaxAge,
    validYearRange,
    years,
    yearRangeLength,
    yearRangeClamp,
    yearRangeEdge,
    yearRangeToStr,
    yearRangeBoundsCheck,
    withdrawalStartYear,
    yourOrYourPartners,
    params,
  }
}

extendTPAWParams.yearRangeEdge = (
  yearRange: YearRange,
  edge: 'start' | 'end'
) =>
  'start' in yearRange && edge === 'start'
    ? yearRange.start
    : 'end' in yearRange && edge === 'end'
    ? yearRange.end
    : null

extendTPAWParams.years = (() => {
  const person = (person: 'person1' | 'person2') => ({
    lastWorkingYear: {type: 'namedAge', person, age: 'lastWorkingYear'} as Year,
    retirement: {type: 'namedAge', person, age: 'retirement'} as Year,
    max: {type: 'namedAge', person: 'person1', age: 'max'} as Year,
    numericAge: (age: number) =>
      ({type: 'numericAge', person: 'person1', age} as Year),
  })
  return {
    now: {type: 'now'} as Year,
    person,
    person1: person('person1'),
    person2: person('person2'),
  }
})()
