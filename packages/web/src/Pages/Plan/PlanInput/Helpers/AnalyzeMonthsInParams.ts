import { GlidePath, Month, MonthRange, ValueForMonthRange } from '@tpaw/common'
import _ from 'lodash'
import { PlanParamsExt } from '../../../../TPAWSimulator/PlanParamsExt'
import { noCase } from '../../../../Utils/Utils'
import { planSectionLabel } from './PlanSectionLabel'

export function analyzeMonthsInParams(
  paramsExt: PlanParamsExt,
  monthUpdater?: (month: Month) => Month,
) {
  const { params } = paramsExt
  const {
    wealth,
    adjustmentsToSpending: { extraSpending },
    risk,
  } = params
  const valueForMonthRange = [
    ...wealth.futureSavings.map((x) =>
      _analyzeValueForMonthRange(x, 'future-savings', paramsExt, monthUpdater),
    ),
    ...wealth.retirementIncome.map((x) =>
      _analyzeValueForMonthRange(
        x,
        'income-during-retirement',
        paramsExt,
        monthUpdater,
      ),
    ),
    ...extraSpending.essential.map((x) =>
      _analyzeValueForMonthRange(x, 'extra-spending', paramsExt, monthUpdater),
    ),
    ...extraSpending.discretionary.map((x) =>
      _analyzeValueForMonthRange(x, 'extra-spending', paramsExt, monthUpdater),
    ),
  ]

  const spawGlidePath = _analyzeGlidePath(
    risk.spawAndSWR.allocation,
    'strategy',
    'assetAllocationForSPAW',
    paramsExt,
    monthUpdater,
  )

  const glidePath = _.compact([
    // Condition only on returning the results, not on running the month updater.
    params.advanced.strategy === 'SPAW' ? spawGlidePath : undefined,
  ])
  return { valueForMonthRange, glidePath }
}

const _analyzeGlidePath = (
  glidePath: GlidePath,
  section: 'strategy',
  location: 'assetAllocationForSPAW',
  paramsExt: PlanParamsExt,
  monthUpdater?: (month: Month) => Month,
) => {
  const { glidePathIntermediateValidated } = paramsExt
  if (monthUpdater) {
    glidePath.intermediate.forEach((x) => (x.month = monthUpdater(x.month)))
  }
  const analyzed = glidePathIntermediateValidated(glidePath.intermediate)

  const sectionLabel = planSectionLabel(section)
  const usesPerson2 = glidePath.intermediate.some((x) => _usesPerson2(x.month))
  return {
    section,
    sectionLabel,
    glidePath,
    analyzed,
    location,
    usesPerson2,
    usesRetirement: (person: 'person1' | 'person2') =>
      glidePath.intermediate.some((x) => _usesRetirement(x.month, person)),
  }
}

const _analyzeValueForMonthRange = (
  entry: ValueForMonthRange,
  section: 'future-savings' | 'income-during-retirement' | 'extra-spending',
  paramsExt: PlanParamsExt,
  monthUpdater?: (month: Month) => Month,
) => {
  const { monthRange, label } = entry
  const { validMonthRangeAsMFN, monthRangeBoundsCheck, params } = paramsExt

  const sectionLabel = planSectionLabel(section)
  if (monthUpdater) {
    switch (monthRange.type) {
      case 'startAndEnd':
        monthRange.start = monthUpdater(monthRange.start)
        monthRange.end = monthUpdater(monthRange.end)
        break
      case 'startAndNumMonths':
        monthRange.start = monthUpdater(monthRange.start)
        break
      case 'endAndNumMonths':
        monthRange.end = monthUpdater(monthRange.end)
        break
      default:
        noCase(monthRange)
    }
  }
  const boundsCheck = monthRangeBoundsCheck(
    monthRange,
    validMonthRangeAsMFN(section),
  )

  const doesSomeEdgeMonth = (
    fn: (x: Month) => boolean,
    monthRange: MonthRange,
  ) => {
    switch (monthRange.type) {
      case 'startAndEnd':
        return fn(monthRange.start) || fn(monthRange.end)
      case 'startAndNumMonths':
        return fn(monthRange.start)
      case 'endAndNumMonths':
        return fn(monthRange.end)
      default:
        noCase(monthRange)
    }
  }
  return {
    boundsCheck,
    section,
    sectionLabel,
    entry,
    usesPerson2: doesSomeEdgeMonth(_usesPerson2, monthRange),
    useRetirement: (person: 'person1' | 'person2') =>
      doesSomeEdgeMonth((x) => _usesRetirement(x, person), monthRange),
  }
}

const _usesPerson2 = (month: Month) =>
  (month.type === 'namedAge' || month.type === 'numericAge') &&
  month.person === 'person2'

const _usesRetirement = (month: Month, person: 'person1' | 'person2') =>
  month.type === 'namedAge' &&
  month.person === person &&
  (month.age === 'retirement' || month.age === 'lastWorkingMonth')
