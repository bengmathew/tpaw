import {
  TPAWParams,
  ValueForYearRange,
  Year,
  YearRange,
} from '../../../../TPAWSimulator/TPAWParams'
import {extendTPAWParams} from '../../../../TPAWSimulator/TPAWParamsExt'
import {noCase} from '../../../../Utils/Utils'
import {mapAllValueForYearsRangesInParams} from './MapAllValueForYearRangesInParams'
import {paramsInputLabel} from './ParamsInputLabel'

type _Type = 'future-savings' | 'income-during-retirement' | 'extra-spending'
export const paramsInputValidate = (params: TPAWParams, type: _Type) => {
  return mapAllValueForYearsRangesInParams(params, paramsInputValidateYearRange)
    .filter(x => x.sectionType === type)
    .every(x => x.boundsCheck.start === 'ok' && x.boundsCheck.end === 'ok')
}

export const paramsInputValidateYearRange = (
  entry: ValueForYearRange,
  sectionType: _Type,
  params: TPAWParams,
  yearUpdater?: (year: Year) => Year
) => {
  const {yearRange, label} = entry
  const {asYFN, validYearRange, yearRangeBoundsCheck} = extendTPAWParams(params)
  const boundsCheck = yearRangeBoundsCheck(
    yearRange,
    validYearRange(sectionType)
  )

  const sectionLabel = paramsInputLabel(sectionType)
  const usesPerson2 = (year: Year) =>
    (year.type === 'namedAge' || year.type === 'numericAge') &&
    year.person === 'person2'
  const usesRetirement = (year: Year, person: 'person1' | 'person2') =>
    (year.type === 'namedAge' || year.type === 'numericAge') &&
    year.person === person &&
    (year.age === 'retirement' || year.age === 'lastWorkingYear')
  if (yearUpdater) {
    switch (yearRange.type) {
      case 'startAndEnd':
        yearRange.start = yearUpdater(yearRange.start)
        yearRange.end = yearUpdater(yearRange.end)
        break
      case 'startAndNumYears':
        yearRange.start = yearUpdater(yearRange.start)
        break
      case 'endAndNumYears':
        yearRange.end = yearUpdater(yearRange.end)
        break
      default:
        noCase(yearRange)
    }
  }
  const doesSomeEdgeYear = (fn: (x: Year) => boolean, yearRange: YearRange) => {
    switch (yearRange.type) {
      case 'startAndEnd':
        return fn(yearRange.start) || fn(yearRange.end)
      case 'startAndNumYears':
        return fn(yearRange.start)
      case 'endAndNumYears':
        return fn(yearRange.end)
      default:
        noCase(yearRange)
    }
  }
  return {
    boundsCheck,
    sectionType,
    sectionLabel,
    entry,
    usesPerson2: doesSomeEdgeYear(usesPerson2, yearRange),
    useRetirement: (person: 'person1' | 'person2') =>
      doesSomeEdgeYear(x => usesRetirement(x, person), yearRange),
  }
}
