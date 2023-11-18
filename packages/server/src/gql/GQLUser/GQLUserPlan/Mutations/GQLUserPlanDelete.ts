import { API, assert } from '@tpaw/common'
import { Clients } from '../../../../Clients.js'
import {
  PrismaTransaction,
  serialTransaction,
} from '../../../../Utils/PrismaTransaction.js'
import { builder } from '../../../builder.js'

const Input = builder.inputType('UserPlanDeleteInput', {
  fields: (t) => ({
    userId: t.string(),
    planId: t.string(),
  }),
})

builder.mutationField('userPlanDelete', (t) =>
  t.prismaField({
    type: 'User',
    authScopes: (_, args, ctx) => ctx.user?.id === args.input.userId,
    args: { input: t.arg({ type: Input }) },
    resolve: async (query, _, { input }) => {
      const { userId, planId } = API.UserPlanDelete.check(input).force()

      await serialTransaction(
        async (tx) => await userPlanDelete(tx, userId, planId),
      )
      return Clients.prisma.user.findUniqueOrThrow({
        ...query,
        where: { id: userId },
      })
    },
  }),
)

export const userPlanDelete = async (
  tx: PrismaTransaction,
  userId: string,
  planId: string,
) => {
  const plan = await tx.planWithHistory.findUniqueOrThrow({
    where: { userId_planId: { userId, planId } },
  })
  assert(!plan.isMain)

  await tx.planWithHistory.delete({
    where: { userId_planId: { userId, planId } },
  })
}
