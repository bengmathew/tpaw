import { Clients } from '../../Clients.js'
import { builder } from '../builder.js'

builder.queryField('linkBasedPlan', (t) =>
  t.prismaField({
    type: 'LinkBasedPlan',
    args: { linkId: t.arg({ type: 'ID' }) },
    nullable: true,
    resolve: async (query, _, args) =>
      await Clients.prisma.linkBasedPlan.findUnique({
        ...query,
        where: { id: `${args.linkId}` },
      }),
  }),
)

builder.prismaObject('LinkBasedPlan', {
  fields: (t) => ({
    id: t.exposeID('id'),
    createdAt: t.float({ resolve: ({ createdAt }) => createdAt.getTime() }),
    params: t.string({ resolve: ({ params }) => JSON.stringify(params) }),
  }),
})
