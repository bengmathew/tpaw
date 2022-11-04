import { API, PlanParams } from '@tpaw/common'
import { JSONGuard } from 'json-guard'
import { Clients } from '../../Clients.js'
import { builder } from '../builder.js'

const Input = builder.inputType('SetUserPlanInput', {
  fields: (t) => ({ userId: t.id(), params: t.string() }),
})

builder.mutationField('setUserPlan', (t) =>
  t.prismaField({
    type: 'User',
    authScopes: (_, args, ctx) => ctx.user?.id === args.input.userId,
    args: { input: t.arg({ type: Input }) },
    resolve: async (query, _, { input }) => {
      let guard: JSONGuard<
        Omit<typeof input, 'params'> & { params: PlanParams }
      > = API.SetUserPlan.check
      const { userId, params } = API.SetUserPlan.check(input).force()

      const now = new Date()
      return await Clients.prisma.user.update({
        ...query,
        where: { id: userId },
        data: {
          plan: {
            upsert: {
              create: { createdAt: now, modifiedAt: now, params },
              update: { modifiedAt: now, params },
            },
          },
        },
      })
    },
  }),
)
