import { API } from '@tpaw/common'
import cryptoRandomString from 'crypto-random-string'
import { Clients } from '../../Clients.js'
import { builder } from '../builder.js'

const Input = builder.inputType('CreateLinkBasedPlanInput', {
  fields: (t) => ({ params: t.string() }),
})

builder.mutationField('createLinkBasedPlan', (t) =>
  t.prismaField({
    type: 'LinkBasedPlan',
    args: { input: t.arg({ type: Input }) },
    resolve: async (query, _, { input }) => {
      const { params } = API.CreateLinkBasedPlan.check(input).force()
      return await Clients.prisma.linkBasedPlan.create({
        ...query,
        data: {
          id: cryptoRandomString({ length: 32, type: 'alphanumeric' }),
          createdAt: new Date(),
          params: params,
        },
      })
    },
  }),
)
