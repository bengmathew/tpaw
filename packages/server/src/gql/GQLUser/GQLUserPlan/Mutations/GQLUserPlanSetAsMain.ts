import { API, fGet } from '@tpaw/common'
import { assert } from 'console'
import { Clients } from '../../../../Clients.js'
import {
  PrismaTransaction,
  serialTransaction,
} from '../../../../Utils/PrismaTransaction.js'
import { builder } from '../../../builder.js'

const Input = builder.inputType('UserPlanSetAsMainInput', {
  fields: (t) => ({
    userId: t.string(),
    planId: t.string(),
  }),
})

builder.mutationField('userPlanSetAsMain', (t) =>
  t.prismaField({
    type: 'User',
    authScopes: (_, args, ctx) => ctx.user?.id === args.input.userId,
    args: { input: t.arg({ type: Input }) },
    resolve: async (query, __, { input }) => {
      const { userId, planId } = API.UserPlanSetAsMain.check(input).force()

      await serialTransaction(
        async (tx) => await userPlanSetAsMain(tx, userId, planId),
      )
      return Clients.prisma.user.findUniqueOrThrow({
        ...query,
        where: { id: userId },
      })
    },
  }),
)

export const userPlanSetAsMain = async (
  tx: PrismaTransaction,
  userId: string,
  planId: string,
) => {
  const plans = await tx.planWithHistory.findMany({
    where: { userId, isMain: true },
  })
  assert(plans.length === 1)
  const currMainPlan = fGet(plans[0])

  const now = new Date()
  await tx.planWithHistory.update({
    where: { userId_planId: { userId, planId: currMainPlan.planId } },
    data: {
      isMain: false,
      sortTime: now,
    },
  })
  await tx.planWithHistory.update({
    where: { userId_planId: { userId, planId } },
    data: {
      isMain: true,
      sortTime: now,
    },
  })
}
