import _ from 'lodash'
import {pluralize} from '../Utils/Pluralize'
import {SimpleRange} from '../Utils/SimpleRange'
import {assert, noCase} from '../Utils/Utils'
import {GlidePath, Person, PlanParams, Year, YearRange} from '@tpaw/common'

export type PlanParamsExt = ReturnType<typeof extendPlanParams>
export const extendPlanParams = (params: PlanParams) => {
  const {years, yearRangeEdge} = extendPlanParams

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

  const maxLastWorkingYear = params.people.withPartner
    ? maxYear(years.person1.lastWorkingYear, years.person2.lastWorkingYear)
    : years.person1.lastWorkingYear

  const minRetirement = params.people.withPartner
    ? minYear(years.person1.retirement, years.person2.retirement)
    : years.person1.retirement

  const maxMaxAge = params.people.withPartner
    ? maxYear(years.person1.max, years.person2.max)
    : years.person1.max
  const numYears = getNumYears(params)
  

  const yearRangeBoundsCheck = (x: YearRange, bounds: SimpleRange) => {
    const inRange = (edge: 'start' | 'end') =>
      _.inRange(asYFN(x)[edge], bounds.start, bounds.end + 1)
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

  const glidePathIntermediateValidated = (
    intermediateIn: GlidePath['intermediate']
  ) => {
    const withoutIssues = _.sortBy(
      intermediateIn.map(x => ({...x, yearAsYFN: asYFN(x.year)})),
      x => x.yearAsYFN
    )

    const intermediate = withoutIssues.map((x, i) => ({
      ...x,
      issue:
        x.yearAsYFN <= 0
          ? ('before' as const)
          : x.yearAsYFN >= numYears - 1
          ? ('after' as const)
          : i !== 0 && withoutIssues[i - 1].yearAsYFN === x.yearAsYFN
          ? ('duplicate' as const)
          : ('none' as const),
    }))

    return intermediate
  }

  const validYearRange = (
    type: 'future-savings' | 'income-during-retirement' | 'extra-spending'
  ) =>
    type === 'future-savings'
      ? asYFN({start: years.now, end: maxLastWorkingYear})
      : type === 'income-during-retirement'
      ? asYFN({start: minRetirement, end: maxMaxAge})
      : type === 'extra-spending'
      ? asYFN({start: years.now, end: maxMaxAge})
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

  const numRetirementYears = numYears- asYFN(withdrawalStartYear)
  const withdrawalsStarted = asYFN(withdrawalStartYear) === 0

  const yourOrYourPartners = (person: 'person1' | 'person2') =>
    person === 'person1' ? 'your' : "your partner's"

  const yearToStr = (year: Year) => {
    const withPrefix = ({person}: {person: 'person1' | 'person2'}, x: string) =>
      params.people.withPartner ? `${yourOrYourPartners(person)} ${x}` : x
    switch (year.type) {
      case 'now':
        return 'now'
      case 'numericAge':
        return withPrefix(year, `age ${year.age}`)
      case 'namedAge': {
        switch (year.age) {
          case 'lastWorkingYear':
            return withPrefix(year, 'last working year')
          case 'retirement':
            return withPrefix(year, 'retirement')
          case 'max':
            return withPrefix(year, 'max age')
          default:
            noCase(year)
        }
      }
      default:
        noCase(year)
    }
  }

  // ----------------------
  // YEAR RANGE  TO STRING
  // -----------------
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
        last.value = `${last.value}${value.startsWith('.') ? '' : ' '}${value}`
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
    numYears,
    pickPerson,
    asYearsFromNow,
    asYFN,
    minYear,
    maxYear,
    maxLastWorkingYear,
    minRetirement,
    maxMaxAge,
    validYearRange,
    glidePathIntermediateValidated,
    years,
    yearRangeLength,
    yearRangeClamp,
    yearRangeEdge,
    yearRangeToStr,
    yearToStr,
    yearRangeBoundsCheck,
    withdrawalStartYear,
    withdrawalsStarted,
    numRetirementYears,
    yourOrYourPartners,
    params,
  }
}

extendPlanParams.yearRangeEdge = (
  yearRange: YearRange,
  edge: 'start' | 'end'
) =>
  'start' in yearRange && edge === 'start'
    ? yearRange.start
    : 'end' in yearRange && edge === 'end'
    ? yearRange.end
    : null

extendPlanParams.years = (() => {
  const person = (person: 'person1' | 'person2') => ({
    lastWorkingYear: {type: 'namedAge', person, age: 'lastWorkingYear'} as Year,
    retirement: {type: 'namedAge', person, age: 'retirement'} as Year,
    max: {type: 'namedAge', person, age: 'max'} as Year,
    numericAge: (age: number) => ({type: 'numericAge', person, age} as Year),
  })
  return {
    now: {type: 'now'} as Year,
    person,
    person1: person('person1'),
    person2: person('person2'),
  }
})()

export const getNumYears = (params: PlanParams) => {
  const forPerson = (person: Person) =>
    person.ages.max - person.ages.current + 1
  return Math.max(
    forPerson(params.people.person1),
    params.people.withPartner ? forPerson(params.people.person2) : 0
  )
}

export const getWithdrawalStartAsYFN = (params: PlanParams) => {
  const person = params.people.withPartner
    ? params.people.withdrawalStart
    : 'person1'
  const {ages} = (() => {
    if (person === 'person1') return params.people.person1
    assert(params.people.withPartner)
    return params.people.person2
  })()
  return ages.type === 'retired' ? 0: ages.retirement - ages.current
}
