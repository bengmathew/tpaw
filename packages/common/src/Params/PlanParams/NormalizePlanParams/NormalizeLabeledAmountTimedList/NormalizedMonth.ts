import { Month } from '../../PlanParams'
import { NormalizedAges } from '../NormalizeAges'
import { SimpleRange } from '../../../../Misc/SimpleRange'
import { block, fGet, noCase } from '../../../../Utils'
import { yourOrYourPartners } from '../../../../Misc/YourOrYourPartners'

export type NormalizedMonthNotInThePast = {
  type: 'normalizedMonth'
  isInThePast: false
  asMFN: number
  baseValue: Month
  validRangeAsMFN: {
    includingLocalConstraints: SimpleRange | null
    excludingLocalConstraints: SimpleRange
  }
  errorMsg: string | null
}
export type NormalizedMonthInThePast = {
  type: 'normalizedMonth'
  isInThePast: true
  validRangeAsMFN: {
    includingLocalConstraints: SimpleRange | null
    excludingLocalConstraints: SimpleRange
  }
  // In the past is an error, so errorMsg cannot be null.
  errorMsg: string
}

// We want to rewrite month as 'now' if === 0 and month is 'retired'. We have
// to remove the reference to 'retired' because if retirement month is 0, we
// will elide retired date to 'retiredWithNoRetirementDateSpecified', and
// then 'retired' will no longer be resolvable.
const _elideRetirementMonthIfZero = (
  x: NormalizedMonthNotInThePast,
  nowMonth: Extract<Month, { type: 'now' }>,
): NormalizedMonthNotInThePast => ({
  ...x,
  baseValue:
    x.asMFN === 0 &&
    x.baseValue.type === 'namedAge' &&
    x.baseValue.age === 'retirement'
      ? nowMonth
      : x.baseValue,
})

export const getNormalizedMonthNotInThePast = (
  asMFN: number,
  baseValue: Month,
  nowMonth: Extract<Month, { type: 'now' }>,
  validRangeAsMFN: {
    includingLocalConstraints: SimpleRange | null
    excludingLocalConstraints: SimpleRange
  },
  process:
    | { type: 'rangeCheckAndSquishRangeForAge'; ages: NormalizedAges }
    | { type: 'raw'; errorMsg: string | null },
): NormalizedMonthNotInThePast => {
  const data = {
    asMFN,
    baseValue,
    validRangeAsMFN,
  }
  return process.type === 'raw'
    ? _elideRetirementMonthIfZero(
        {
          type: 'normalizedMonth',
          isInThePast: false,
          ...data,
          errorMsg: process.errorMsg,
        },
        nowMonth,
      )
    : normalizedMonthRangeCheckAndSquishRangeForAge(data, process.ages, nowMonth)
}

export const normalizedMonthRangeCheckAndSquishRangeForAge = (
  {
    asMFN,
    baseValue,
    validRangeAsMFN,
  }: {
    asMFN: number
    baseValue: Month
    validRangeAsMFN: {
      includingLocalConstraints: SimpleRange | null
      excludingLocalConstraints: SimpleRange
    }
  },
  ages: {
    person1: { maxAge: { asMFN: number } }
    person2: { maxAge: { asMFN: number } } | null
  },
  nowMonth: Extract<Month, { type: 'now' }>,
): NormalizedMonthNotInThePast => {
  const ageRangeInfo = block(() => {
    switch (baseValue.type) {
      case 'calendarMonth':
      case 'now':
      case 'namedAge':
        return { hasAgeRange: false } as const
      case 'numericAge':
        return {
          hasAgeRange: true,
          ageRangeAsMFN: {
            start: 0,
            end: fGet(ages[baseValue.person]).maxAge.asMFN,
          },
          personType: baseValue.person,
        } as const
      default:
        noCase(baseValue)
    }
  })
  return _elideRetirementMonthIfZero(
    {
      type: 'normalizedMonth',
      isInThePast: false,
      asMFN,
      baseValue,
      validRangeAsMFN: {
        includingLocalConstraints: ageRangeInfo.hasAgeRange
          ? SimpleRange.Closed.intersection(
              validRangeAsMFN.includingLocalConstraints,
              ageRangeInfo.ageRangeAsMFN,
            )
          : validRangeAsMFN.includingLocalConstraints,
        excludingLocalConstraints: validRangeAsMFN.excludingLocalConstraints,
      },
      // Order of checks is important.
      errorMsg: !SimpleRange.Closed.isIn(
        asMFN,
        validRangeAsMFN.includingLocalConstraints,
      )
        ? normalizedMonthErrorMsg.outOfRange
        : ageRangeInfo.hasAgeRange &&
            !SimpleRange.Closed.isIn(asMFN, ageRangeInfo.ageRangeAsMFN)
          ? normalizedMonthErrorMsg.pastMaxAge[ageRangeInfo.personType]
          : null,
    },
    nowMonth,
  )
}

export const normalizedMonthErrorMsg = {
  outOfRange: 'Out of range',
  pastMaxAge: {
    person1: `Past ${yourOrYourPartners('person1')} max age`,
    person2: `Past ${yourOrYourPartners('person2')} max age`,
  },
}
