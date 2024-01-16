import {
  JSONGuard,
  array,
  bounded,
  chain,
  failure,
  gte,
  integer,
  json,
  nullable,
  number,
  object,
  string,
  success,
} from 'json-guard'
import _ from 'lodash'
import { Guards } from './Guards'
import { nonPlanParamsBackwardsCompatibleGuard } from './Params/NonPlanParams/NonPlanParams'
import {
  planParamsBackwardsCompatibleGuard,
  planParamsBackwardsUpToTimestampCompatibleGuard,
  planParamsMigrate,
} from './Params/PlanParams/PlanParams'
import {
  PlanParamsChangeAction,
  planParamsChangeActionGuard,
  planParamsChangeActionGuardCurrent,
} from './Params/PlanParams/PlanParamsChangeAction'
import { fGet } from './Utils'

export namespace API {
  // Update this if the client needs to forced to update.
  export const version = '2'
  // Update this if the you want to inform the user there is a new version 
  // available, if they want to update.
  export const clientVersion = '1'
  const { uuid, ianaTimezoneName } = Guards

  const trimmed: JSONGuard<string, string> = (x) =>
    x.trim().length === x.length
      ? success(x)
      : failure('String is not trimmed.')

  const nonEmpty: JSONGuard<string, string> = (x) =>
    x.length > 0 ? success(x) : failure('Empty string.')

  const email: JSONGuard<string> = chain(string, trimmed, (x) => {
    const EMAIL_REGEX = /^[^@]+@([^@]+\.[^@]+)$/
    const DNS_REGEX =
      /^(([a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9\-]*[a-zA-Z0-9])\.)*([A-Za-z0-9]|[A-Za-z0-9][A-Za-z0-9\-]*[A-Za-z0-9])$/

    const emailMatch = EMAIL_REGEX.exec(x)
    if (emailMatch === null || !emailMatch[1])
      return failure('Email is invalid.')
    if (!DNS_REGEX.test(emailMatch[1]))
      return failure('DNS part of email is invalid')
    return success(x)
  })

  const userId = chain(string, bounded(100))
  const planLabel = chain(string, nonEmpty, trimmed, bounded(500))

  const isFirstChange = (change: PlanParamsChangeAction) =>
    change.type === 'start' ||
    change.type === 'startCopiedFromBeforeHistory' ||
    change.type === 'startCutByClient' ||
    change.type === 'startFromURL'
  const planForCreate = chain(
    object({
      planParamsHistory: chain(
        array(
          object({
            id: uuid,
            params: chain(string, json, planParamsBackwardsCompatibleGuard),
            change: chain(string, json, planParamsChangeActionGuard),
          }),
          10000,
        ),
        (planParamsHistory) => {
          if (planParamsHistory.length === 0)
            return failure('planParamsHistory must have at least one item.')
          const first = fGet(_.first(planParamsHistory))

          if (!isFirstChange(first.change))
            return failure('Not a valid first change.')

          {
            const index = planParamsHistory.findIndex(
              (x, i) => i !== 0 && isFirstChange(x.change),
            )
            if (index !== -1)
              return failure(`Not valid change at index ${index}.`)
          }
          const migrated = planParamsHistory.map((x) => {
            try {
              return planParamsMigrate(x.params)
            } catch (e) {
              return null
            }
          })
          {
            const index = migrated.findIndex((x) => x === null)
            if (index !== -1)
              return failure(`Params at index ${index} is not migratable.`)
          }

          if (
            !_.compact(migrated).every(
              (x, i) =>
                i === migrated.length - 1 ||
                x.timestamp < fGet(migrated[i + 1]).timestamp,
            )
          ) {
            return failure(`Timestamps are not in order.`)
          }

          return success(planParamsHistory)
        },
      ),
      reverseHeadIndex: chain(number, integer, gte(0)),
    }),
    (x) => {
      if (x.reverseHeadIndex > x.planParamsHistory.length - 1)
        return failure('reverseHeadIndex is out of range.')
      return success(x)
    },
  )

  export namespace SendSignInEmail {
    export const guards = { email, dest: string }
    export const check = object(guards)
  }

  export namespace UserPlanCreate {
    export type Plan = {
      planParamsHistory: {
        id: string
        params: string
        change: string
      }[]
      reverseHeadIndex: number
    }
    export const parts = {
      label: planLabel,
    }
    export const check = (x: { userId: string; label: string; plan: Plan }) =>
      object({
        userId,
        label: parts.label,
        plan: planForCreate,
      })(x)
  }

  export namespace UserMergeFromClient {
    export const parts = {
      linkPlan: object({
        label: planLabel,
        plan: planForCreate,
      }),
    }
    export const check = (x: {
      userId: string
      guestPlan?: UserPlanCreate.Plan | null
      linkPlan?: { label: string; plan: UserPlanCreate.Plan } | null
      nonPlanParams?: string | null
    }) =>
      object({
        userId,
        guestPlan: nullable(planForCreate),
        linkPlan: nullable(parts.linkPlan),
        nonPlanParams: nullable(
          chain(string, json, nonPlanParamsBackwardsCompatibleGuard),
        ),
      })(x)
  }

  export namespace UserPlanReset {
    export const check = (x: {
      userId: string
      planId: string
      lastSyncAt: number
      ianaTimezoneName: string
    }) =>
      object({
        userId,
        planId: uuid,
        lastSyncAt: chain(number, integer),
        ianaTimezoneName,
      })(x)
  }

  export namespace UserPlanCopy {
    export const parts = {
      label: planLabel,
    }
    export const check = (x: {
      userId: string
      planId: string
      label: string
      cutAfterId?: string | null
    }) =>
      object({
        userId,
        planId: uuid,
        label: parts.label,
        cutAfterId: nullable(uuid),
      })(x)
  }

  export namespace UserPlanSync {
    export const check = (x: {
      userId: string
      planId: string
      lastSyncAt: number
      cutAfterId: string
      add: {
        id: string
        params: string
        change: string
      }[]
      reverseHeadIndex: number
    }) =>
      object({
        userId,
        planId: uuid,
        lastSyncAt: chain(number, integer),
        cutAfterId: uuid,
        add: chain(
          array(
            object({
              id: uuid,
              // Allow backward compatible guard so we can support older clients.
              // Eg. when server is updated, but user has not refreshed client.
              // But need at at least timestamp.
              params: chain(
                string,
                json,
                planParamsBackwardsUpToTimestampCompatibleGuard,
              ),
              change: chain(string, json, planParamsChangeActionGuardCurrent),
            }),
            10000,
          ),
          (add) => {
            {
              const index = add.findIndex((x) => isFirstChange(x.change))
              if (index !== -1)
                return failure(`Not valid change at index ${index}.`)
            }

            if (
              !_.compact(add).every(
                (x, i) =>
                  i === add.length - 1 ||
                  x.params.timestamp < fGet(add[i + 1]).params.timestamp,
              )
            ) {
              return failure(`Timestamps are not in order.`)
            }

            return success(add)
          },
        ),
        reverseHeadIndex: chain(number, integer, gte(0)),
      })(x)
  }

  export namespace UserPlanUpdate {
    export const parts = {
      setLabel: nullable(planLabel),
    }
    export const check = (x: {
      userId: string
      planId: string
      setLabel?: string | null
    }) =>
      object({
        userId,
        planId: uuid,
        setLabel: parts.setLabel,
      })(x)
  }

  export namespace UserPlanDelete {
    export const check = (x: { userId: string; planId: string }) =>
      object({
        userId,
        planId: uuid,
      })(x)
  }

  export namespace UserPlanSetAsMain {
    export const check = (x: { userId: string; planId: string }) =>
      object({
        userId,
        planId: uuid,
      })(x)
  }

  export namespace UserSetNonPlanParams {
    export const check = (x: {
      userId: string
      lastUpdatedAt: number
      nonPlanParams: string
    }) =>
      object({
        userId,
        lastUpdatedAt: chain(number, integer),
        nonPlanParams: chain(
          string,
          json,
          // Allow backward compatible guard so we can support older clients.
          // Eg. when server is updated, but user has not refreshed client.
          nonPlanParamsBackwardsCompatibleGuard,
        ),
      })(x)
  }

  export namespace UserPlan {
    export namespace PlanParamsPreBase {
      export const check = (x: {
        baseTimestamp: number
        baseId: string
        ianaTimezoneName: string
      }) =>
        object({
          baseTimestamp: chain(number, integer),
          baseId: uuid,
          ianaTimezoneName,
        })(x)
    }
  }

  export namespace GeneratePDFReport {
    export const check = (x: {
      url: string
      auth?: string | null
      viewportWidth: number
      viewportHeight: number
      devicePixelRatio: number
    }) =>
      object({
        url: string,
        auth: nullable(string),
        viewportWidth: chain(number, integer, gte(1)),
        viewportHeight: chain(number, integer, gte(1)),
        devicePixelRatio: chain(number, integer, gte(1)),
      })(x)

    export type Input = InputTypeFromCheck<typeof check>
  }

  export namespace CreateLinkBasedPlan {
    export const check = (x: { params: string }) =>
      object({
        params: chain(string, json, planParamsBackwardsCompatibleGuard),
      })(x)
  }
}

type InputTypeFromCheck<T> = T extends JSONGuard<infer U, infer V> ? U : never
