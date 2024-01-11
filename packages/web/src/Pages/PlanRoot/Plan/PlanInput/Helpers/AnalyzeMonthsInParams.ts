import { GlidePath, Month, ValueForMonthRange } from '@tpaw/common'
import _ from 'lodash'
import { PlanParamsExtended } from '../../../../../UseSimulator/ExtentPlanParams'
import { noCase } from '../../../../../Utils/Utils'
import { planSectionLabel } from './PlanSectionLabel'

// Uses MasterListOfMonths.

export function analyzeMonthsInParams(
  planParamsExt: PlanParamsExtended,
  opts: { type: 'asVisible' } | { type: 'includeNotVisible' },
) {
  const { planParams } = planParamsExt
  const {
    wealth,
    risk,
    adjustmentsToSpending: { extraSpending },
  } = planParams
  const valueForMonthRange = [
    ..._.values(wealth.futureSavings)
      .sort((a, b) => a.sortIndex - b.sortIndex)
      .map((x) =>
        _analyzeValueForMonthRange(x, 'future-savings', planParamsExt, opts),
      ),
    ..._.values(wealth.incomeDuringRetirement)
      .sort((a, b) => a.sortIndex - b.sortIndex)
      .map((x) =>
        _analyzeValueForMonthRange(
          x,
          'income-during-retirement',
          planParamsExt,
          opts,
        ),
      ),
    ..._.values(extraSpending.essential)
      .sort((a, b) => a.sortIndex - b.sortIndex)
      .map((x) =>
        _analyzeValueForMonthRange(x, 'extra-spending', planParamsExt, opts),
      ),
    ..._.values(extraSpending.discretionary)
      .sort((a, b) => a.sortIndex - b.sortIndex)
      .map((x) =>
        _analyzeValueForMonthRange(x, 'extra-spending', planParamsExt, opts),
      ),
  ]

  const glidePath = _.compact([
    opts.type === 'includeNotVisible' ||
    planParams.advanced.strategy === 'SPAW' ||
    planParams.advanced.strategy === 'SWR'
      ? _analyzeGlidePath(
          risk.spawAndSWR.allocation,
          'risk',
          planParamsExt,
          opts.type,
        )
      : undefined,
  ])
  return { valueForMonthRange, glidePath }
}

const _analyzeGlidePath = (
  glidePath: GlidePath,
  section: 'risk',
  planParamsExt: PlanParamsExtended,
  type: 'asVisible' | 'includeNotVisible',
) => {
  const { glidePathIntermediateValidated } = planParamsExt

  const analyzed = glidePathIntermediateValidated(
    glidePath.intermediate,
  ).filter((x) => (type === 'includeNotVisible' ? true : x.issue !== 'before'))

  const sectionLabel = planSectionLabel(section)
  const usesPerson2 = analyzed.some((x) => _usesPerson2(x.month))
  return {
    section,
    sectionLabel,
    glidePath,
    analyzed,
    usesPerson2,
    usesRetirement: (person: 'person1' | 'person2') =>
      analyzed.some((x) => _usesRetirement(x.month, person)),
  }
}

const _analyzeValueForMonthRange = (
  entry: ValueForMonthRange,
  section: 'future-savings' | 'income-during-retirement' | 'extra-spending',
  planParamsExt: PlanParamsExtended,
  opts: { type: 'asVisible' } | { type: 'includeNotVisible' },
) => {
  const { monthRange: rangeUnclamped } = entry
  const { validMonthRangeAsMFN, monthRangeBoundsCheck, clampMonthRangeToNow } =
    planParamsExt

  const sectionLabel = planSectionLabel(section)
  const range =
    opts.type === 'asVisible'
      ? clampMonthRangeToNow(rangeUnclamped)
      : rangeUnclamped

  const boundsCheck = range
    ? monthRangeBoundsCheck(range, validMonthRangeAsMFN(section))
    : null

  const doesSomeEdgeMonth = (fn: (x: Month) => boolean) => {
    if (!range) return false
    switch (range.type) {
      case 'startAndEnd':
        return fn(range.start) || fn(range.end)
      case 'startAndNumMonths':
        return fn(range.start)
      case 'endAndNumMonths':
        return fn(range.end)
      default:
        noCase(range)
    }
  }
  return {
    boundsCheck,
    section,
    sectionLabel,
    entry,
    usesPerson2: doesSomeEdgeMonth(_usesPerson2),
    useRetirement: (person: 'person1' | 'person2') =>
      doesSomeEdgeMonth((x) => _usesRetirement(x, person)),
  }
}

const _usesPerson2 = (month: Month) =>
  (month.type === 'namedAge' || month.type === 'numericAge') &&
  month.person === 'person2'

const _usesRetirement = (month: Month, person: 'person1' | 'person2') =>
  month.type === 'namedAge' &&
  month.person === person &&
  (month.age === 'retirement' || month.age === 'lastWorkingMonth')
