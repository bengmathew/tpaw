import { Month, block, fGet, letIn, noCase } from '@tpaw/common'
import { SimpleRange } from '../../../Utils/SimpleRange'
import { NormalizedAges } from '../NormalizeAges'
import { yourOrYourPartners } from '../../../Utils/YourOrYourPartners'

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
  // In the past is and error, so errorMsg cannot be null.
  errorMsg: string
}

export const getNormalizedMonthNotInThePast = (
  asMFN: number,
  baseValue: Month,
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
    ? {
        type: 'normalizedMonth',
        isInThePast: false,
        ...data,
        errorMsg: process.errorMsg,
      }
    : normalizedMonthRangeCheckAndSquishRangeForAge(data, process.ages)
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
  return {
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
  }
}

export const normalizedMonthErrorMsg = {
  outOfRange: 'Out of range',
  pastMaxAge: {
    person1: `Past ${yourOrYourPartners('person1')} max age`,
    person2: `Past ${yourOrYourPartners('person2')} max age`,
  },
}
