import { Clients } from '../../Clients.js'
import { builder } from '../builder.js'

export const PothosPlanAndUserResult = builder
  .objectRef<{ type: 'PlanAndUserResult'; userId: string; planId: string }>(
    'PlanAndUserResult',
  )
  .implement({
    fields: (t) => ({
      plan: t.prismaField({
        type: 'PlanWithHistory',
        resolve: async (query, { userId, planId }) =>
          await Clients.prisma.planWithHistory.findUniqueOrThrow({
            ...query,
            where: { userId_planId: { userId, planId } },
          }),
      }),
      user: t.prismaField({
        type: 'User',
        resolve: async (query, { userId }) =>
          await Clients.prisma.user.findUniqueOrThrow({
            ...query,
            where: { id: userId },
          }),
      }),
    }),
  })
