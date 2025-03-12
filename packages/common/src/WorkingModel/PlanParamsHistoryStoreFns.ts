import jsonpatch, { Operation } from 'fast-json-patch'
import { cloneJSON } from '../Misc/CloneJSON.js'
import {
  PlanParams,
  planParamsMigrate,
  SomePlanParams,
} from '../Params/PlanParams/PlanParams.js'
import { PlanParamsChangeAction } from '../Params/PlanParams/PlanParamsChangeAction/PlanParamsChangeAction.js'
import { assert, noCase } from '../Utils.js'

// TODO: Consider mergeing with PlanParamsHistoryFns. Check which of that is
// still used.
export namespace PlanParamsHistoryStoreFns {
  export type Item = {
    id: string
    change: PlanParamsChangeAction
    planParamsUnmigrated: SomePlanParams
    planParams: PlanParams
  }

  export type ItemStored = {
    id: string
    timestamp: number
    change: PlanParamsChangeAction
    reverseDiff: Operation[]
  }

  export const fromStored = (
    endingParams: SomePlanParams,
    planParamsHistoryReversed: ItemStored[],
  ) => {
    const result: Item[] = []
    const currParams = cloneJSON(endingParams)
    planParamsHistoryReversed.forEach((x, i) => {
      const planParamsUnmigrated = cloneJSON(currParams)
      const planParams = planParamsMigrate(currParams)
      assert(planParams.timestamp === x.timestamp)
      assert(
        result.length === 0 ||
          planParams.timestamp < result[i - 1]!.planParams.timestamp,
      )
      result.push({
        id: x.id,
        change: x.change,
        planParamsUnmigrated,
        planParams,
      })
      jsonpatch.applyPatch(currParams, x.reverseDiff)
    })
    result.reverse() // In-place reverse.
    return result
  }

  export const toStored = (
    currEndingParams:
      | { type: 'forAdd'; planParamsUnmigrated: SomePlanParams }
      | { type: 'forCreate' },
    newChanges: Item[],
  ) => {
    // Ensure changes are in the correct order.
    {
      const timestamps = (
        currEndingParams.type === 'forAdd'
          ? [currEndingParams, ...newChanges]
          : newChanges
      ).map((x) =>
        'timestamp' in x.planParamsUnmigrated
          ? x.planParamsUnmigrated.timestamp
          : planParamsMigrate(x.planParamsUnmigrated).timestamp,
      )
      assert(timestamps.every((x, i) => i === 0 || x > timestamps[i - 1]!))
    }

    let curr =
      currEndingParams.type === 'forAdd'
        ? currEndingParams.planParamsUnmigrated
        : currEndingParams.type === 'forCreate'
          ? {}
          : noCase(currEndingParams)

    return newChanges.map((x) => {
      const reverseDiff = jsonpatch.compare(x.planParamsUnmigrated, curr)
      curr = x.planParamsUnmigrated
      return {
        planParamsChangeId: x.id,
        timestamp: new Date(x.planParams.timestamp),
        reverseDiff,
        change: x.change,
      }
    })
  }
}
