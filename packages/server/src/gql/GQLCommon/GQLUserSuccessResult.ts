import { Clients } from '../../Clients.js'
import { builder } from '../builder.js'

builder.objectType('UserSuccessResult', {
  fields: (t) => ({
    user: t.prismaField({
      type: 'User',
      resolve: async (query, { userId }) =>
        await Clients.prisma.user.findUniqueOrThrow({
          ...query,
          where: { id:userId },
        }),
    }),
  }),
})
