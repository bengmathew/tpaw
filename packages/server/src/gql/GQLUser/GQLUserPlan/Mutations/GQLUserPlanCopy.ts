import { API, fGet, getSlug } from '@tpaw/common'
import { assert } from 'console'
import * as uuid from 'uuid'
import { serialTransaction } from '../../../../Utils/PrismaTransaction.js'
import { PothosPlanAndUserResult } from '../../../GQLCommon/GQLPlanAndUserResult.js'
import { builder } from '../../../builder.js'
import { userPlanSync } from './GQLUserPlanSync.js'

const Input = builder.inputType('UserPlanCopyInput', {
  fields: (t) => ({
    userId: t.string(),
    planId: t.string(),
    label: t.string(),
    cutAfterId: t.string({ required: false }),
  }),
})

builder.mutationField('userPlanCopy', (t) =>
  t.field({
    type: PothosPlanAndUserResult,
    authScopes: (_, args, ctx) => ctx.user?.id === args.input.userId,
    args: { input: t.arg({ type: Input }) },
    resolve: async (_, { input }) => {
      const { userId, planId, label, cutAfterId } =
        API.UserPlanCopy.check(input).force()

      const newPlanId = uuid.v4()
      await serialTransaction(async (tx) => {
        const currPlans = await tx.planWithHistory.findMany({
          where: { userId },
        })
        assert(currPlans.length <= 100)

        const src = await tx.planWithHistory.findUniqueOrThrow({
          where: { userId_planId: { userId, planId } },
          include: { paramsChangeHistory: true },
        })

        const now = new Date()
        await tx.planWithHistory.create({
          data: {
            planId: newPlanId,
            isMain: false,
            userId,
            addedToServerAt: now,
            sortTime: now,
            lastSyncAt: now,
            label,
            slug: getSlug(
              label,
              currPlans.map((x) => x.slug),
            ),
            resetCount: 0,

            endingParams: fGet(src.endingParams),
            paramsChangeHistory: {
              createMany: {
                data: src.paramsChangeHistory.map((x) => ({
                  // Intentionally not generating a new uuid here, because we
                  // want to  ensure that the planParamsChangeId referred to
                  // other change items (eg. is
                  // currentPortfolioBalance.updatedAtId) are still valid.
                  planParamsChangeId: x.planParamsChangeId,
                  timestamp: x.timestamp,
                  reverseDiff: fGet(x.reverseDiff),
                  change: fGet(x.change),
                })),
              },
            },
            reverseHeadIndex: src.reverseHeadIndex,
          },
        })

        if (cutAfterId) {
          // 0 is ok for reverseHeadIndex, because if cutting, we expect
          // cutAfterId to be earlier that the current reverseHeadIndex.
          await userPlanSync(tx, userId, planId, cutAfterId, [], 0)
        }
      })
      return { type: 'PlanAndUserResult' as const, userId, planId: newPlanId }
    },
  }),
)
