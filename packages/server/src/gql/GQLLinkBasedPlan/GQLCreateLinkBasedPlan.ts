import { API, Params } from '@tpaw/common'
import cryptoRandomString from 'crypto-random-string'
import { JSONGuard } from 'json-guard'
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
      let guard: JSONGuard<
        Omit<typeof input, 'params'> & { params: Params }
      > = API.CreateLinkBasedPlan.check
      const { params } = guard(input).force()
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
