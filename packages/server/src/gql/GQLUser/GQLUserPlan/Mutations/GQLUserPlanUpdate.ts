import { API, getSlug } from '@tpaw/common'
import { Clients } from '../../../../Clients.js'
import { serialTransaction } from '../../../../Utils/PrismaTransaction.js'
import { builder } from '../../../builder.js'

const Input = builder.inputType('UserPlanUpdateInput', {
  fields: (t) => ({
    userId: t.string(),
    planId: t.string(),
    setLabel: t.string({ required: false }),
  }),
})

builder.mutationField('userPlanUpdate', (t) =>
  t.prismaField({
    type: 'PlanWithHistory',
    authScopes: (_, args, ctx) => ctx.user?.id === args.input.userId,
    args: { input: t.arg({ type: Input }) },
    resolve: async (query, _, { input }) => {
      const { userId, planId, setLabel } =
        API.UserPlanUpdate.check(input).force()

      await serialTransaction(async (tx) => {
        if (setLabel) {
          const currSlugs = (
            await tx.planWithHistory.findMany({
              where: { userId, planId: { not: planId } },
            })
          ).map((x) => x.slug)
          await tx.planWithHistory.update({
            ...query,
            where: { userId_planId: { userId, planId } },
            data: {
              label: setLabel ?? undefined,
              slug: getSlug(setLabel, currSlugs),
            },
          })
        }
      })
      return await Clients.prisma.planWithHistory.findUniqueOrThrow({
        ...query,
        where: { userId_planId: { userId, planId } },
      })
    },
  }),
)
