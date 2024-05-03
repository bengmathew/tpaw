import * as Sentry from '@sentry/nextjs'
import {
  Guards,
  PlanParams,
  PlanParamsChangeAction,
  SomePlanParams,
  block,
  planParamsBackwardsCompatibleGuard,
  planParamsChangeActionGuard,
} from '@tpaw/common'
import cloneJSON from 'fast-json-clone'
import jsonpatch, { Operation } from 'fast-json-patch'
import {
  array,
  chain,
  constant,
  failure,
  gte,
  integer,
  json,
  number,
  object,
  success,
} from 'json-guard'
import { JSONGuard } from 'json-guard/dist/definition'
import * as uuid from 'uuid'

export type PlanFileData = {
  v: 1
  convertedToFilePlanAtTimestamp: number
  lastSavedTimestamp: number
  planParamsHistory: {
    id: string
    change: PlanParamsChangeAction
    params: SomePlanParams
  }[]
  reverseHeadIndex: number
}
export const PLAN_FILE_EXTENSION = '.tpaw.txt'
const prefix = `
HOW TO OPEN THIS FILE
---------------------

This file contains a plan from TPAW Planner. 
You can open this plan from the plan menu at:

https://tpawplanner.com/plan


---- PLAN DATA STARTS HERE ----
`

export namespace PlanFileDataFns {
  type Stored = {
    v: 1
    convertedToFilePlanAtTimestamp: number
    lastSavedTimestamp: number
    planParamsHistory: {
      id: string
      change: PlanParamsChangeAction
      diff: Operation[]
    }[]
    reverseHeadIndex: number
  }

  const guard: JSONGuard<PlanFileData, string> = chain(
    (x) => {
      let startIndex = x.indexOf('{')
      if (startIndex === -1) {
        return failure('Could not find opening "{"')
      }
      return success(x.slice(startIndex))
    },
    json,
    object({
      v: constant(1),
      convertedToFilePlanAtTimestamp: chain(number, integer, gte(0)),
      lastSavedTimestamp: chain(number, integer, gte(0)),
      planParamsHistory: chain(
        (x) => {
          let curr = {} as any
          const guard = array((x) => {
            const objectCheck = object({
              id: Guards.uuid,
              change: planParamsChangeActionGuard,
              diff: array((x) => success(x as Operation), 10000),
            })(x)
            if (objectCheck.error) return objectCheck
            const { id, change, diff } = objectCheck.value
            try {
              jsonpatch.applyPatch(curr, diff)
            } catch (e) {
              return failure(`Error patching planParamsHistory`)
            }
            const paramsCheck = planParamsBackwardsCompatibleGuard(
              cloneJSON(curr),
            )
            if (paramsCheck.error) return paramsCheck
            return success({ id, change, params: paramsCheck.value })
          }, 10000)
          return guard(x)
        },
        (x) =>
          x.length > 0
            ? success(x)
            : fail('planParamsHistory must have at least one item'),
      ),
      reverseHeadIndex: chain(number, integer, gte(0)),
    }),
  )

  export const getNew = (planParams: PlanParams): PlanFileData => ({
    v: 1,
    convertedToFilePlanAtTimestamp: planParams.timestamp,
    lastSavedTimestamp: planParams.timestamp,
    planParamsHistory: [
      block(() => {
        const change: PlanParamsChangeAction = { type: 'start', value: null }
        return { id: uuid.v4(), change, params: planParams }
      }),
    ],
    reverseHeadIndex: 0,
  })
  export const getFromLink = (planParams: SomePlanParams): PlanFileData => {
    const now = Date.now()
    return {
      v: 1,
      convertedToFilePlanAtTimestamp: now,
      lastSavedTimestamp: now,
      planParamsHistory: [
        block(() => {
          const change: PlanParamsChangeAction = { type: 'start', value: null }
          return { id: uuid.v4(), change, params: planParams }
        }),
      ],
      reverseHeadIndex: 0,
    }
  }

  export const download = (filename: string, data: PlanFileData) => {
    const stored: Stored = {
      v: data.v,
      convertedToFilePlanAtTimestamp: data.convertedToFilePlanAtTimestamp,
      lastSavedTimestamp: data.lastSavedTimestamp,
      planParamsHistory: data.planParamsHistory.map((curr, i) => ({
        id: curr.id,
        change: curr.change,
        diff: jsonpatch.compare(
          i === 0 ? {} : data.planParamsHistory[i - 1].params,
          curr.params,
        ),
      })),
      reverseHeadIndex: data.reverseHeadIndex,
    }
    const str = `${prefix}${JSON.stringify(stored)}`
    guard(str).force()
    const url = URL.createObjectURL(new Blob([str], { type: 'text/plain' }))
    const link = document.createElement('a')
    link.href = url
    link.download = filename
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  }

  export const open = async (file: File): Promise<PlanFileData | null> => {
    const str = await file.text()
    const check = guard(str)
    if (check.error) {
      console.dir(check.message)
      Sentry.captureException(new Error(`Error opening file: ${check.message}`))
    }
    return check.error ? null : check.value
  }
  export const labelFromFilename = (filename: string | null): string | null =>
    filename ? filename.split('.')[0] : null
}
