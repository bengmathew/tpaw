import { assert, fGet } from '@tpaw/common'
import { Clients } from '../../Clients.js'
import { builder } from '../builder.js'

builder.queryField('user', (t) =>
  t.prismaField({
    type: 'User',
    args: { userId: t.arg({ type: 'ID' }) },
    resolve: async (query, _, { userId }) =>
      await Clients.prisma.user.findUniqueOrThrow({
        ...query,
        where: { id: `${userId}` },
      }),
  }),
)

export const PothosUser = builder.prismaObject('User', {
  authScopes: (user, context) => context.user?.id === user.id,
  fields: (t) => ({
    id: t.exposeID('id'),
    nonPlanParams: t.string({
      resolve: ({ nonPlanParams }) => JSON.stringify(nonPlanParams),
    }),
    nonPlanParamsLastUpdatedAt: t.float({
      resolve: (x) => x.nonPlanParamsLastUpdatedAt.getTime(),
    }),
    plans: t.prismaField({
      type: ['PlanWithHistory'],
      resolve: async (query, { id: userId }) =>
        await Clients.prisma.planWithHistory.findMany({
          ...query,
          where: { userId },
          orderBy: { sortTime: 'desc' },
        }),
    }),
    plan: t.prismaField({
      type: 'PlanWithHistory',
      args: {
        slug: t.arg.string({ required: false }),
        planId: t.arg.string({ required: false }),
      },
      resolve: async (query, { id: userId }, { slug, planId }) => {
        const plans = await Clients.prisma.planWithHistory.findMany({
          ...query,
          where: planId
            ? { userId, planId }
            : slug
            ? { userId, slug }
            : { userId, isMain: true },
        })
        assert(plans.length === 1)
        return fGet(plans[0])
      },
    }),
  }),
})
