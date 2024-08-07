import { API, SomePlanParams } from '@tpaw/common'
import * as uuid from 'uuid'
import {
  PrismaTransaction,
  serialTransaction,
} from '../../../../Utils/PrismaTransaction.js'
import { concurrentChangeError } from '../../../../impl/Common/ConcurrentChangeError.js'
import { PothosPlanAndUserResult } from '../../../GQLCommon/GQLPlanAndUserResult.js'
import { builder } from '../../../builder.js'
import { patchPlanParams } from '../PatchPlanParams.js'

const Input = builder.inputType('UserPlanResetInput', {
  fields: (t) => ({
    userId: t.string(),
    planId: t.string(),
    lastSyncAt: t.float(),
    planParams: t.string(),
  }),
})

builder.mutationField('userPlanReset', (t) =>
  t.field({
    type: builder.unionType('UserPlanResetResult', {
      types: ['ConcurrentChangeError', PothosPlanAndUserResult],
      resolveType: (value) => value.type,
    }),
    authScopes: (_, args, ctx) => ctx.user?.id === args.input.userId,
    args: { input: t.arg({ type: Input }) },
    resolve: async (__, { input }) => {
      const { userId, planId, lastSyncAt, planParams } =
        API.UserPlanReset.check(input).force()

      return await serialTransaction(async (tx) => {
        const startingPlan = await tx.planWithHistory.findUniqueOrThrow({
          where: { userId_planId: { userId, planId } },
        })
        if (startingPlan.lastSyncAt.getTime() !== lastSyncAt)
          return concurrentChangeError

        await userPlanReset(tx, userId, planId, planParams)
        return { type: 'PlanAndUserResult' as const, planId, userId }
      })
    },
  }),
)

export const userPlanReset = async (
  tx: PrismaTransaction,
  userId: string,
  planId: string,
  planParams: SomePlanParams,
) => {
  const now = new Date()
  const endingParams = planParams

  await tx.planWithHistory.update({
    where: { userId_planId: { userId, planId } },
    data: {
      sortTime: now,
      lastSyncAt: now,
      resetCount: { increment: 1 },
      paramsChangeHistory: {
        deleteMany: {},
        createMany: {
          data: patchPlanParams.generate({ type: 'forCreate' }, [
            {
              id: uuid.v4(),
              params: endingParams,
              change: { type: 'start', value: null },
            },
          ]),
        },
      },
      endingParams,
      reverseHeadIndex: 0,
    },
  })
}
