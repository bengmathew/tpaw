import _ from 'lodash'
import {
  GlidePath,
  ValueForYearRange,
  Year,
  YearRange,
} from '../../../../TPAWSimulator/TPAWParams'
import {TPAWParamsExt} from '../../../../TPAWSimulator/TPAWParamsExt'
import {noCase} from '../../../../Utils/Utils'
import {paramsInputLabel} from './ParamsInputLabel'

export function analyzeYearsInParams(
  paramsExt: TPAWParamsExt,
  yearUpdater?: (year: Year) => Year
) {
  const {params} = paramsExt
  const {savings, retirementIncome, withdrawals, targetAllocation} = params
  const valueForYearRange = [
    ...savings.map(x =>
      _analyzeValueForYearRange(x, 'future-savings', paramsExt, yearUpdater)
    ),
    ...retirementIncome.map(x =>
      _analyzeValueForYearRange(
        x,
        'income-during-retirement',
        paramsExt,
        yearUpdater
      )
    ),
    ...withdrawals.essential.map(x =>
      _analyzeValueForYearRange(x, 'extra-spending', paramsExt, yearUpdater)
    ),
    ...withdrawals.discretionary.map(x =>
      _analyzeValueForYearRange(x, 'extra-spending', paramsExt, yearUpdater)
    ),
  ]

  const spawGlidePath = _analyzeGlidePath(
    targetAllocation.regularPortfolio.forSPAWAndSWR,
    'compare-strategies',
    'assetAllocationForSPAW',
    paramsExt,
    yearUpdater
  )

  const glidePath = _.compact([
    // Condition only on returning the results, not on running the year updater.
    params.strategy === 'SPAW' ? spawGlidePath : undefined,
  ])
  return {valueForYearRange, glidePath}
}

const _analyzeGlidePath = (
  glidePath: GlidePath,
  section: 'compare-strategies',
  location: 'assetAllocationForSPAW',
  paramsExt: TPAWParamsExt,
  yearUpdater?: (year: Year) => Year
) => {
  const {glidePathIntermediateValidated} = paramsExt
  if (yearUpdater) {
    glidePath.intermediate.forEach(x => (x.year = yearUpdater(x.year)))
  }
  const analyzed = glidePathIntermediateValidated(glidePath.intermediate)

  const sectionLabel = paramsInputLabel(section)
  const usesPerson2 = glidePath.intermediate.some(x => _usesPerson2(x.year))
  return {
    section,
    sectionLabel,
    glidePath,
    analyzed,
    location,
    usesPerson2,
    usesRetirement: (person: 'person1' | 'person2') =>
      glidePath.intermediate.some(x => _usesRetirement(x.year, person)),
  }
}

const _analyzeValueForYearRange = (
  entry: ValueForYearRange,
  section: 'future-savings' | 'income-during-retirement' | 'extra-spending',
  paramsExt: TPAWParamsExt,
  yearUpdater?: (year: Year) => Year
) => {
  const {yearRange, label} = entry
  const {validYearRange, yearRangeBoundsCheck, params} = paramsExt

  const sectionLabel = paramsInputLabel(section)
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
  const boundsCheck = yearRangeBoundsCheck(yearRange, validYearRange(section))

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
    section,
    sectionLabel,
    entry,
    usesPerson2: doesSomeEdgeYear(_usesPerson2, yearRange),
    useRetirement: (person: 'person1' | 'person2') =>
      doesSomeEdgeYear(x => _usesRetirement(x, person), yearRange),
  }
}

const _usesPerson2 = (year: Year) =>
  (year.type === 'namedAge' || year.type === 'numericAge') &&
  year.person === 'person2'
const _usesRetirement = (year: Year, person: 'person1' | 'person2') =>
  (year.type === 'namedAge' || year.type === 'numericAge') &&
  year.person === person &&
  (year.age === 'retirement' || year.age === 'lastWorkingYear')
