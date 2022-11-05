import { Clients } from '../../Clients.js'
import { builder } from '../builder.js'

builder.queryField('user', (t) =>
  t.prismaField({
    type: 'User',
    args: { userId: t.arg({ type: 'ID' }) },
    resolve: async (query, _, args) =>
      await Clients.prisma.user.findUniqueOrThrow({
        ...query,
        where: { id: `${args.userId}` },
      }),
  }),
)

builder.prismaObject('User', {
  select: { id: true },
  authScopes: (user, context) => context.user?.id === user.id,
  fields: (t) => ({
    id: t.exposeID('id'),
    plan: t.relation('plan', { nullable: true }),
  }),
})
