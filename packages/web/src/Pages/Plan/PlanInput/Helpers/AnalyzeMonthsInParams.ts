import {
  GlidePath,
  Month,
  PlanParams,
  ValueForMonthRange,
} from '@tpaw/common'
import _ from 'lodash'
import { ParamsExtended } from '../../../../TPAWSimulator/ExtentParams'
import { noCase } from '../../../../Utils/Utils'
import { planSectionLabel } from './PlanSectionLabel'

export function analyzeMonthsInParams(
  plan: PlanParams,
  paramsExtended: ParamsExtended,
  opts:
    | { type: 'asVisible' }
    | {
        type: 'raw'
        monthInRangeUpdater?: (month: Month) => Month
        glidePathUpdater?: (x: GlidePath) => GlidePath
      },
) {
  const {
    wealth,
    risk,
    adjustmentsToSpending: { extraSpending },
  } = plan
  const valueForMonthRange = [
    ...wealth.futureSavings.map((x) =>
      _analyzeValueForMonthRange(x, 'future-savings', paramsExtended, opts),
    ),
    ...wealth.retirementIncome.map((x) =>
      _analyzeValueForMonthRange(
        x,
        'income-during-retirement',
        paramsExtended,
        opts,
      ),
    ),
    ...extraSpending.essential.map((x) =>
      _analyzeValueForMonthRange(x, 'extra-spending', paramsExtended, opts),
    ),
    ...extraSpending.discretionary.map((x) =>
      _analyzeValueForMonthRange(x, 'extra-spending', paramsExtended, opts),
    ),
  ]

  if (opts.type === 'raw' && opts.glidePathUpdater) {
    risk.spawAndSWR.allocation = opts.glidePathUpdater(
      risk.spawAndSWR.allocation,
    )
  }

  const glidePath = _.compact([
    opts.type === 'raw' ||
    plan.advanced.strategy === 'SPAW' ||
    plan.advanced.strategy === 'SWR'
      ? _analyzeGlidePath(
          risk.spawAndSWR.allocation,
          'risk',
          paramsExtended,
          opts.type,
        )
      : undefined,
  ])
  return { valueForMonthRange, glidePath }
}

const _analyzeGlidePath = (
  glidePath: GlidePath,
  section: 'risk',
  paramsExt: ParamsExtended,
  type: 'asVisible' | 'raw',
) => {
  const { glidePathIntermediateValidated } = paramsExt

  const analyzed = glidePathIntermediateValidated(
    glidePath.intermediate,
  ).filter((x) => (type === 'raw' ? true : x.issue !== 'before'))

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
  paramsExt: ParamsExtended,
  opts:
    | { type: 'asVisible' }
    | { type: 'raw'; monthInRangeUpdater?: (month: Month) => Month },
) => {
  const { monthRange: rangeUnclamped } = entry
  const { validMonthRangeAsMFN, monthRangeBoundsCheck, clampMonthRangeToNow } =
    paramsExt

  if (opts.type === 'raw' && opts.monthInRangeUpdater) {
    switch (rangeUnclamped.type) {
      case 'startAndEnd':
        rangeUnclamped.start = opts.monthInRangeUpdater(rangeUnclamped.start)
        rangeUnclamped.end = opts.monthInRangeUpdater(rangeUnclamped.end)
        break
      case 'startAndNumMonths':
        rangeUnclamped.start = opts.monthInRangeUpdater(rangeUnclamped.start)
        break
      case 'endAndNumMonths':
        rangeUnclamped.end = opts.monthInRangeUpdater(rangeUnclamped.end)
        break
      default:
        noCase(rangeUnclamped)
    }
  }

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
