import { API, SomeNonPlanParams } from '@tpaw/common'
import {
  PrismaTransaction,
  serialTransaction,
} from '../../../Utils/PrismaTransaction.js'
import { concurrentChangeError } from '../../../impl/Common/ConcurrentChangeError.js'
import { getUserSuccessResult } from '../../../impl/Common/UserSuccessResult.js'
import { builder } from '../../builder.js'

const Input = builder.inputType('UserSetNonPlanParamsInput', {
  fields: (t) => ({
    userId: t.string(),
    lastUpdatedAt: t.float(),
    nonPlanParams: t.string(),
  }),
})

builder.mutationField('userSetNonPlanParams', (t) =>
  t.field({
    type: builder.unionType('UserSetNonPlanParamsResult', {
      types: ['ConcurrentChangeError', 'UserSuccessResult'],
      resolveType: (value) => value.type,
    }),
    authScopes: (_, args, ctx) => ctx.user?.id === args.input.userId,
    args: { input: t.arg({ type: Input }) },
    resolve: async (_, { input }) => {
      const { userId, lastUpdatedAt, nonPlanParams } =
        API.UserSetNonPlanParams.check(input).force()
      return await serialTransaction(async (tx) => {
        const startingUser = await tx.user.findUniqueOrThrow({
          where: { id: userId },
        })
        if (startingUser.nonPlanParamsLastUpdatedAt.getTime() !== lastUpdatedAt)
          return concurrentChangeError
        await userSetNonPlanParams(tx, userId, nonPlanParams)
        return getUserSuccessResult(userId)
      })
    },
  }),
)

export const userSetNonPlanParams = async (
  tx: PrismaTransaction,
  userId: string,
  nonPlanParams: SomeNonPlanParams,
) => {
  await tx.user.update({
    where: { id: userId },
    data: {
      nonPlanParams: nonPlanParams,
      nonPlanParamsLastUpdatedAt: new Date(),
    },
  })
}
