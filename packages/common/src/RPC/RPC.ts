import { JSONGuard } from 'json-guard'
import * as z from 'zod'
import {
  planParamsBackwardsCompatibleGuard,
  planParamsMigrate,
  SomePlanParams,
} from '../Params/PlanParams/PlanParams'
import {
  PlanParamsChangeAction,
  planParamsChangeActionGuard,
  planParamsChangeActionGuardCurrent,
} from '../Params/PlanParams/PlanParamsChangeAction/PlanParamsChangeAction'
import { fGet, letIn } from '../Utils'

export namespace RPC {
  const userId = z.string().trim().min(1).max(100)
  const ianaTimezoneName = letIn(
    // From: https://github.com/moment/luxon/issues/353#issuecomment-1262828949
    Intl.supportedValuesOf('timeZone'),
    (timeZoneNames) =>
      z
        .string()
        .refine((v) => timeZoneNames.includes(v), 'Invalid IANA timezone name'),
  )

  export type PlanParamsHistoryItem = {
    id: string
    change: PlanParamsChangeAction
    timestamp: number
    reverseDiff: unknown
  }

  export type PlanWithoutParams = {
    planId: string
    isMain: boolean
    slug: string
    label: string | null
    addedToServerAt: number
    sortTime: number
    lastSyncAt: number
    resetCount: number
    endingPlanParams: SomePlanParams
    reverseHeadIndex: number
  }

  export type PlanWithParams = PlanWithoutParams & {
    planParamsHistoryPostBase: PlanParamsHistoryItem[]
  }

  export type PlanPatch = PlanWithoutParams & {
    planParamsHistoryPostBase: {
      cutAfterId: string
      append: PlanParamsHistoryItem[]
    }
  }

  export type InitialModel = {
    user: {
      userId: string
    }
    plans: (PlanWithoutParams | PlanWithParams)[]
  }

  export type PlanTransaction =
    Args<'syncAndGetPatches'>['plans'][number]['transactions'][number]
  export type PlanUpdateAction = Extract<
    PlanTransaction,
    { type: 'update' }
  >['actions'][number]

  const zodFromJSONGuard = <T, F>(guard: JSONGuard<T, F>) =>
    z.custom<F>().transform((v, ctx) => {
      const check = guard(v)
      if (check.error) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: check.message,
        })
        return z.NEVER
      }
      return check.value
    })

  const planPatchArgs = {
    planParamsHistoryEnding: z
      .array(
        z.object({
          id: z.string().uuid(),
          timestamp: z.number().int().positive(),
        }),
      )
      .min(1)
      .max(5000),
  }

  const trimmed = (x: z.ZodString) =>
    x.refine((v) => v.trim().length === v.length, 'Not trimmed.')
  const planLabel = trimmed(z.string().min(1).max(500))

  const _validators = {
    getInitialLocalModel: {
      args: z.object({
        userId,
        minPlanParamsPostBaseSize: z.number().int().positive().max(500),
        additionalPlanId: z.string().uuid().nullable(),
      }),
      result: null as unknown as InitialModel,
    },
    getInitialPlan: {
      args: z.object({
        userId,
        planId: z.string().uuid(),
      }),
      result: null as unknown as PlanWithParams | null,
    },
    getPlanParamsHistoryPreBase: {
      args: z.object({
        userId,
        planId: z.string().uuid(),
        baseTimestamp: z.number().int().positive(),
      }),
      result: null as unknown as PlanParamsHistoryItem[],
    },
    syncAndGetPatches: {
      args: z.object({
        ianaTimezoneName,
        plans: z.array(
          z.object({
            userId,
            planId: z.string().uuid(),
            patchArgsIfNotDeleted: z.object({ ...planPatchArgs }).nullable(),
            transactions: z
              .array(
                z.discriminatedUnion('type', [
                  z.object({
                    type: z.literal('create'),
                    label: planLabel.nullable(),
                    destPlanId: z.string().uuid(),
                    planParamsHistory: z
                      .array(
                        z.object({
                          id: z.string().uuid(),
                          change: zodFromJSONGuard(planParamsChangeActionGuard),
                          planParamsUnmigrated: zodFromJSONGuard(
                            planParamsBackwardsCompatibleGuard,
                          ),
                        }),
                      )
                      .min(1)
                      .max(100000)
                      .refine((x) => {
                        const timestamps = x.map(
                          (x) =>
                            planParamsMigrate(x.planParamsUnmigrated).timestamp,
                        )
                        return timestamps.every(
                          (x, i) => i === 0 || x > fGet(timestamps[i - 1]),
                        )
                      }, 'Timetamps are not in order.'),
                    // FEATURE: If refine returned a ZodObject (as it should
                    // when 4 releases:
                    // https://github.com/colinhacks/zod/issues/2474#issuecomment-2101077363)
                    // we could use it to enforce that headId is present in the
                    // history. As of now, using .refine() would break
                    // discriminated unions.
                    headId: z.string().uuid(),
                  }),
                  z.object({
                    type: z.literal('delete'),
                  }),
                  z.object({
                    type: z.literal('copy'),
                    destPlanId: z.string().uuid(),
                  }),
                  z.object({
                    type: z.literal('reset'),
                    changeId: z.string().uuid(),
                  }),
                  z.object({
                    type: z.literal('update'),
                    currentResetCount: z.number().int().gte(0),
                    actions: z
                      .array(
                        z.discriminatedUnion('type', [
                          z.object({
                            type: z.literal('setLabel'),
                            label: planLabel,
                          }),
                          z.object({
                            type: z.literal('planParamChange'),
                            // We require the changeId to come from the client
                            // instead of generating one of the server because
                            // setHead refers to an id, and has to be consistent
                            // with the one on the client.
                            changeId: z.string().uuid(),
                            planParamsChangeAction: zodFromJSONGuard(
                              planParamsChangeActionGuardCurrent,
                            ),
                          }),
                          z.object({
                            type: z.literal('setHead'),
                            targetId: z.string().uuid(),
                          }),
                        ]),
                      )
                      .min(1)
                      .max(1000),
                  }),
                ]),
              )
              .min(1)
              .max(500)
              .refine((x) => {
                const lastCreateIndex = x.findLastIndex(
                  (x) => x.type === 'create',
                )
                return lastCreateIndex === 0 || lastCreateIndex === -1
              }, 'create must be the first transaction')
              .refine((x) => {
                const firstDeleteIndex = x.findIndex((x) => x.type === 'delete')
                return (
                  firstDeleteIndex === -1 || firstDeleteIndex === x.length - 1
                )
              }, 'delete must be the last action'),
          }),
        ),
      }),
      result: null as unknown as {
        plans: {
          planId: string
          conflictDetected: boolean
          patch: PlanPatch | 'unpatchable'
        }[]
      },
    },
    getPlanPatch: {
      args: z.object({
        userId,
        planId: z.string().uuid(),
        ...planPatchArgs,
      }),
      result: null as unknown as PlanPatch | 'unpatchable',
    },
  }

  type _Validators = typeof _validators
  export type MethodName = keyof _Validators
  export type Args<T extends MethodName> = z.infer<_Validators[T]['args']>
  export type Result<T extends MethodName> = _Validators[T]['result']

  export const methodNames = Object.keys(_validators) as MethodName[]

  const _getMainArgs = <T extends keyof _Validators>(str: T) =>
    z.object({
      method: z.literal(str),
      args: _validators[str].args as _Validators[T]['args'],
    })

  export const mainArgs = z.discriminatedUnion('method', [
    _getMainArgs('getInitialLocalModel'),
    _getMainArgs('getPlanParamsHistoryPreBase'),
    _getMainArgs('syncAndGetPatches'),
    _getMainArgs('getPlanPatch'),
  ])
  export type MainArgs = z.infer<typeof mainArgs>
}
