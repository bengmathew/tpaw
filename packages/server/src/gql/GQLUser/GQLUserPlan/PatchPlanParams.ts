import { PlanParamsChange, Prisma } from '@prisma/client'
import {
  PlanParamsChangeAction,
  SomePlanParams,
  assert,
  cloneJSON,
  noCase,
  planParamsMigrate,
} from '@tpaw/common'
import jsonpatch, { Operation } from 'fast-json-patch'
import _ from 'lodash'

// TODO: Deprecated. Moved to common.
export type PlanParamsChangePatched = {
  userId: string
  planId: string
  planParamsChangeId: string
  params: SomePlanParams
  change: PlanParamsChangeAction
  timestamp: number
}

export const patchPlanParams = (
  endingParams: SomePlanParams,
  planParamsHistoryReversed: PlanParamsChange[],
  filterIn: 'all' | ((x: PlanParamsChange, reverseIndex: number) => boolean),
  { checkTimestamps = true }: { checkTimestamps?: boolean } = {},
): PlanParamsChangePatched[] => {
  const filter = filterIn === 'all' ? () => true : filterIn
  const result: (PlanParamsChangePatched | null)[] = []
  const currParams = cloneJSON(endingParams)
  let prevTimestamp = Infinity
  planParamsHistoryReversed.forEach((x, reverseI) => {
    assert(x.timestamp.getTime() < prevTimestamp)
    prevTimestamp = x.timestamp.getTime()
    const item: PlanParamsChangePatched | null = filter(x, reverseI)
      ? {
          userId: x.userId,
          planId: x.planId,
          planParamsChangeId: x.planParamsChangeId,
          params: cloneJSON(currParams),
          change: x.change as PlanParamsChangeAction,
          timestamp: x.timestamp.getTime(),
        }
      : null
    jsonpatch.applyPatch(currParams, x.reverseDiff as unknown as Operation[])
    result.push(item)
  })
  const compacted = _.compact(result).reverse()
  // Sanity check that reverseDiffing is correct.
  const check = (x?: PlanParamsChangePatched) => {
    if (x) assert(planParamsMigrate(x.params).timestamp === x.timestamp)
  }
  if (process.env['NODE_ENV'] === 'development' || checkTimestamps) {
    compacted.forEach(check)
  } else {
    check(_.first(compacted))
    check(_.last(compacted))
  }
  return compacted
}
patchPlanParams.forSingle = (
  endingParams: SomePlanParams,
  planParamsHistoryReversed: PlanParamsChange[],
) => {
  const result = patchPlanParams(
    endingParams,
    planParamsHistoryReversed,
    (__, reverseI) => reverseI === planParamsHistoryReversed.length - 1,
  )
  return _.first(result)
}

patchPlanParams.generate = (
  currLastHistoryItem:
    | { type: 'forAdd'; params: SomePlanParams; timestamp: number }
    | { type: 'forCreate' },
  newChanges: {
    id: string
    params: SomePlanParams
    change: PlanParamsChangeAction
  }[],
) => {
  let [currEndingParams, prevTimestamp] =
    currLastHistoryItem.type === 'forAdd'
      ? [currLastHistoryItem.params, currLastHistoryItem.timestamp]
      : currLastHistoryItem.type === 'forCreate'
        ? [{}, -Infinity]
        : noCase(currLastHistoryItem)

  return newChanges.map((x) => {
    const reverseDiff = jsonpatch.compare(x.params, currEndingParams)
    currEndingParams = x.params
    const timestamp = planParamsMigrate(x.params).timestamp
    assert(timestamp > prevTimestamp)
    prevTimestamp = timestamp
    return {
      planParamsChangeId: x.id,
      timestamp: new Date(timestamp),
      reverseDiff: reverseDiff as unknown as Prisma.InputJsonArray,
      change: x.change,
    }
  })
}
