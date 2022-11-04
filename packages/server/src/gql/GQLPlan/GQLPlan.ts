import { builder } from '../builder.js'

builder.prismaObject('Plan', {
  fields: (t) => ({
    id: t.exposeID('id'),
    createdAt: t.float({ resolve: ({ createdAt }) => createdAt.getTime() }),
    modifiedAt: t.float({ resolve: ({ modifiedAt }) => modifiedAt.getTime() }),
    params: t.string({ resolve: ({ params }) => JSON.stringify(params) }),
  }),
})
