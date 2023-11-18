import { API, PlanParamsChangeAction, assert, block, fGet } from '@tpaw/common'
import _ from 'lodash'
import { Clients } from '../../../Clients.js'
import {
  PrismaTransaction,
  serialTransaction,
} from '../../../Utils/PrismaTransaction.js'
import { builder } from '../../builder.js'
import { PothosPlanWithHistory } from './GQLUserPlan.js'
import {
  PothosUserPlanCreatePlanInput,
  userPlanCreate,
} from './Mutations/GQLUserPlanCreate.js'
import { userPlanDelete } from './Mutations/GQLUserPlanDelete.js'
import { userPlanSetAsMain } from './Mutations/GQLUserPlanSetAsMain.js'

const Input = builder.inputType('UserMergeFromClientInput', {
  fields: (t) => ({
    userId: t.string(),
    guestPlan: t.field({
      type: PothosUserPlanCreatePlanInput,
      required: false,
    }),
    linkPlan: t.field({
      type: builder.inputType('UserMergeFromClientLinkPlanInput', {
        fields: (t) => ({
          label: t.string(),
          plan: t.field({ type: PothosUserPlanCreatePlanInput }),
        }),
      }),
      required: false,
    }),
    nonPlanParams: t.string({ required: false }),
  }),
})

const ReturnType = block(() => {
  return builder
    .objectRef<{
      userId: string
      guestPlanId: string | null
      linkPlanId: string | null
    }>('UserMergeFromClientResult')
    .implement({
      fields: (t) => ({
        guestPlan: t.prismaField({
          type: PothosPlanWithHistory,
          nullable: true,
          resolve: async (query, { userId, guestPlanId }) => {
            if (!guestPlanId) return null
            return await Clients.prisma.planWithHistory.findUniqueOrThrow({
              ...query,
              where: { userId_planId: { userId, planId: guestPlanId } },
            })
          },
        }),
        linkPlan: t.prismaField({
          type: PothosPlanWithHistory,
          nullable: true,
          resolve: async (query, { userId, linkPlanId }) => {
            if (!linkPlanId) return null
            return await Clients.prisma.planWithHistory.findUniqueOrThrow({
              ...query,
              where: { userId_planId: { userId, planId: linkPlanId } },
            })
          },
        }),
      }),
    })
})

// Ref: https://pothos-graphql.dev/docs/plugins/prisma#optimized-queries-without-tprismafield
builder.mutationField('userMergeFromClient', (t) =>
  t.field({
    type: ReturnType,
    authScopes: (_, args, ctx) => ctx.user?.id === args.input.userId,
    args: { input: t.arg({ type: Input }) },
    resolve: async (__, { input }, context, info) => {
      const { userId, guestPlan, linkPlan, nonPlanParams } =
        API.UserMergeFromClient.check(input).force()

      // FEATURE: Don't ignore nonPlanParams
      return serialTransaction(async (tx) => {
        return {
          userId,
          // ---- GUEST PLAN ----
          guestPlanId: await block(async () => {
            if (!guestPlan) return null

            const { currMainPlan, isCurrMainPlanOverwritable } =
              await _isCurrMainPlanOverwriteable(tx, userId)

            if (
              currMainPlan.paramsChangeHistory.length === 1 &&
              guestPlan.planParamsHistory.length === 1 &&
              _.isEqual(
                currMainPlan.endingParams,
                fGet(guestPlan.planParamsHistory[0]).params,
              )
            )
              return null

            const newPlan = await userPlanCreate(
              tx,
              userId,
              `Copied From Browser`,
              guestPlan.planParamsHistory,
              guestPlan.reverseHeadIndex,
              false,
            )
            if (isCurrMainPlanOverwritable) {
              await userPlanSetAsMain(tx, userId, newPlan.planId)
              await userPlanDelete(tx, userId, currMainPlan.planId)
            }
            return newPlan.planId
          }),
          // ---- LINK PLAN ----
          linkPlanId: await block(async () => {
            if (!linkPlan) return null

            const { currMainPlan, isCurrMainPlanOverwritable } =
              await _isCurrMainPlanOverwriteable(tx, userId)

            const newPlan = await userPlanCreate(
              tx,
              userId,
              linkPlan.label,
              linkPlan.plan.planParamsHistory,
              linkPlan.plan.reverseHeadIndex,
              false,
            )
            if (isCurrMainPlanOverwritable) {
              await userPlanSetAsMain(tx, userId, newPlan.planId)
              await userPlanDelete(tx, userId, currMainPlan.planId)
            }
            return newPlan.planId
          }),
        }
      })
    },
  }),
)

const _isCurrMainPlanOverwriteable = async (
  tx: PrismaTransaction,
  userId: string,
) => {
  const currMainPlans = await tx.planWithHistory.findMany({
    where: { userId, isMain: true },
    include: {
      paramsChangeHistory: {
        orderBy: { timestamp: 'asc' },
        take: 2,
      },
    },
  })
  const numPlans = await tx.planWithHistory.count()
  assert(currMainPlans.length === 1)
  const currMainPlan = fGet(currMainPlans[0])

  return {
    currMainPlan,
    isCurrMainPlanOverwritable:
      numPlans === 1 &&
      currMainPlan.paramsChangeHistory.length === 1 &&
      (
        fGet(_.first(currMainPlan.paramsChangeHistory))
          .change as PlanParamsChangeAction
      ).type === 'start',
  }
}
