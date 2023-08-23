import { API, getDefaultPlanParams } from '@tpaw/common'
import * as uuid from 'uuid'
import { Clients } from '../../../../Clients.js'
import { PrismaTransaction } from '../../../../Utils/PrismaTransaction.js'
import { concurrentChangeError } from '../../../../impl/Common/ConcurrentChangeError.js'
import { PothosPlanAndUserResult } from '../../../GQLCommon/GQLPlanAndUserResult.js'
import { builder } from '../../../builder.js'
import { patchPlanParams } from '../PatchPlanParams.js'

const Input = builder.inputType('UserPlanResetInput', {
  fields: (t) => ({
    userId: t.string(),
    planId: t.string(),
    lastSyncAt: t.float(),
    ianaTimezoneName: t.string(),
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
      const { userId, planId, lastSyncAt, ianaTimezoneName } =
        API.UserPlanReset.check(input).force()

      return await Clients.prisma.$transaction(async (tx) => {
        const startingPlan = await tx.planWithHistory.findUniqueOrThrow({
          where: { userId_planId: { userId, planId } },
        })
        if (startingPlan.lastSyncAt.getTime() !== lastSyncAt)
          return concurrentChangeError

        await userPlanReset(tx, userId, planId, ianaTimezoneName)
        return { type: 'PlanAndUserResult' as const, planId, userId }
      })
    },
  }),
)

export const userPlanReset = async (
  tx: PrismaTransaction,
  userId: string,
  planId: string,
  ianaTimezoneName: string,
) => {
  const now = new Date()
  const endingParams = getDefaultPlanParams(now.getTime(), ianaTimezoneName)
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
