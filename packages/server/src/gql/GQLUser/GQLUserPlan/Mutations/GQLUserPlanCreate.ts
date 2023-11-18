import { Prisma } from '@prisma/client'
import Sentry from '@sentry/node'
import {
  API,
  PlanParamsChangeAction,
  SomePlanParams,
  fGet,
  getSlug,
} from '@tpaw/common'
import { assert } from 'console'
import _ from 'lodash'
import * as uuid from 'uuid'
import {
  PrismaTransaction,
  serialTransaction,
} from '../../../../Utils/PrismaTransaction.js'
import { PothosPlanAndUserResult } from '../../../GQLCommon/GQLPlanAndUserResult.js'
import { builder } from '../../../builder.js'
import { patchPlanParams } from '../PatchPlanParams.js'

export const PothosUserPlanCreatePlanInput = builder.inputType(
  'UserPlanCreatePlanInput',
  {
    fields: (t) => ({
      planParamsHistory: t.field({
        type: [
          builder.inputType('UserPlanCreatePlanParamsHistryInput', {
            fields: (t) => ({
              id: t.string(),
              params: t.string(),
              change: t.string(),
            }),
          }),
        ],
      }),
      reverseHeadIndex: t.int(),
    }),
  },
)
const Input = builder.inputType('UserPlanCreateInput', {
  fields: (t) => ({
    userId: t.string(),
    label: t.string(),
    plan: t.field({
      type: PothosUserPlanCreatePlanInput,
    }),
  }),
})

builder.mutationField('userPlanCreate', (t) =>
  t.field({
    type: PothosPlanAndUserResult,
    authScopes: (_, args, ctx) => ctx.user?.id === args.input.userId,
    args: { input: t.arg({ type: Input }) },
    resolve: async (__, { input }, context) => {
      const { userId, label, plan } = API.UserPlanCreate.check(input).force()

      const newPlan = await serialTransaction(
        async (tx) =>
          await userPlanCreate(
            tx,
            userId,
            label,
            plan.planParamsHistory,
            plan.reverseHeadIndex,
            false,
          ),
      )

      // return await Clients.prisma.user.findUniqueOrThrow({
      //   ...query,
      //   where: { id: userId },
      // })

      return {
        type: 'PlanAndUserResult' as const,
        userId,
        planId: newPlan.planId,
      }
    },
  }),
)

export const userPlanCreate = async (
  tx: PrismaTransaction,
  userId: string,
  label: string,
  planParamsHistory: {
    id: string
    params: SomePlanParams
    change: PlanParamsChangeAction
  }[],
  reverseHeadIndex: number,
  isMain: boolean,
) => {
  const currPlans = await tx.planWithHistory.findMany({
    where: { userId },
  })
  assert(currPlans.length <= 500)

  const now = new Date()

  const endingParams = fGet(_.last(planParamsHistory)).params
  const slug = getSlug(
    label,
    currPlans.map((x) => x.slug),
  )
  const createData: Prisma.PlanWithHistoryUncheckedCreateInput = {
    planId: uuid.v4(),
    isMain,
    userId,
    addedToServerAt: now,
    sortTime: now,
    lastSyncAt: now,
    label,
    slug,
    resetCount: 0,
    endingParams,
    paramsChangeHistory: {
      createMany: {
        data: patchPlanParams.generate(
          { type: 'forCreate' },
          planParamsHistory,
        ),
      },
    },
    reverseHeadIndex,
  }
  try {
    return await tx.planWithHistory.create({
      data: createData,
    })
  } catch (e) {
    Sentry.captureMessage(
      JSON.stringify({
        userId,
        createData,
        slugs: {
          before: currPlans.map((x) => x.slug),
          after: (
            await tx.planWithHistory.findMany({
              where: { userId },
            })
          ).map((x) => x.slug),
        },
      }),
    )
    throw e
  }
}
