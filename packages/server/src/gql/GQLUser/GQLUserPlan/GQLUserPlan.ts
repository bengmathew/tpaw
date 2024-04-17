import {
  API,
  PlanParamsHistoryFns,
  SomePlanParams,
  assert,
  fGet,
  planParamsMigrate,
} from '@tpaw/common'
import _ from 'lodash'
import { Clients } from '../../../Clients.js'
import { builder } from '../../builder.js'
import { patchPlanParams } from './PatchPlanParams.js'

const PothosPlanParamsChangePatched = builder.objectType(
  'PlanParamsChangePatched',
  {
    authScopes: (planHistoryItem, context) =>
      context.user?.id === planHistoryItem.userId,
    fields: (t) => ({
      id: t.exposeID('planParamsChangeId'),
      params: t.string({ resolve: ({ params }) => JSON.stringify(params) }),
      change: t.string({ resolve: ({ change }) => JSON.stringify(change) }),
    }),
  },
)

export const PothosPlanWithHistory = builder.prismaObject('PlanWithHistory', {
  authScopes: (plan, context) => context.user?.id === plan.userId,
  fields: (t) => ({
    id: t.exposeID('planId'),
    isMain: t.exposeBoolean('isMain'),
    label: t.exposeString('label', { nullable: true }),
    slug: t.exposeString('slug'),
    addedToServerAt: t.float({ resolve: (x) => x.addedToServerAt.getTime() }),
    sortTime: t.float({ resolve: (x) => x.sortTime.getTime() }),
    lastSyncAt: t.float({ resolve: (x) => x.lastSyncAt.getTime() }),
    reverseHeadIndex: t.exposeInt('reverseHeadIndex'),
    planParamsPostBase: t.field({
      type: [PothosPlanParamsChangePatched],
      args: { targetCount: t.arg.int() },
      resolve: async (
        { planId, userId, endingParams: endingParamsIn, reverseHeadIndex },
        { targetCount },
      ) => {
        const endingParams = endingParamsIn as SomePlanParams
        assert(targetCount >= 0)
        assert(targetCount > reverseHeadIndex)

        const requestedStart = fGet(
          patchPlanParams.forSingle(
            endingParams,
            await Clients.prisma.planParamsChange.findMany({
              where: { planId, userId },
              orderBy: { timestamp: 'desc' },
              take: targetCount,
            }),
          ),
        )
        const startPortfolioBalance = planParamsMigrate(requestedStart.params)
          .wealth.portfolioBalance

        // Go back to minId because that is needed for current portfolio
        // balance estimation.
        const minId =
          !startPortfolioBalance.isDatedPlan ||
          startPortfolioBalance.updatedHere
            ? requestedStart.planParamsChangeId
            : startPortfolioBalance.updatedAtId
        const minTimestamp = (
          await Clients.prisma.planParamsChange.findUniqueOrThrow({
            where: {
              userId_planId_planParamsChangeId: {
                userId,
                planId,
                planParamsChangeId: minId,
              },
            },
          })
        ).timestamp

        const p1 = await Clients.prisma.planParamsChange.findMany({
          where: { planId, userId, timestamp: { gte: minTimestamp } },
          orderBy: { timestamp: 'desc' },
        })
        return patchPlanParams(endingParams, p1, () => true)
      },
    }),

    planParamsPreBase: t.field({
      type: [PothosPlanParamsChangePatched],
      args: {
        baseTimestamp: t.arg.float(),
        baseId: t.arg.string(),
        ianaTimezoneName: t.arg.string(),
      },
      resolve: async ({ planId, userId, endingParams }, args) => {
        const { baseTimestamp, baseId, ianaTimezoneName } =
          API.UserPlan.PlanParamsPreBase.check(args).force()
        // NOTE: Can' filter planParamsHistory until after
        // patchPlanParams() because we need all the reverseDiffs.
        const planParamsHistory =
          await Clients.prisma.planParamsChange.findMany({
            where: { planId, userId },
            orderBy: { timestamp: 'asc' },
          })

        const { idsToDelete } = PlanParamsHistoryFns.filterForHistoryFromStart({
          ianaTimezoneName,
          marketCloses: 'useConservativeGuess',
          planParamsHistory: planParamsHistory.filter(
            (x) => x.timestamp.getTime() <= baseTimestamp,
          ),
        })
        const result = patchPlanParams(
          endingParams as SomePlanParams,
          planParamsHistory.reverse(),
          (x) =>
            !idsToDelete.has(x.planParamsChangeId) &&
            x.timestamp.getTime() <= baseTimestamp,
        )

        const lastEntry = fGet(_.last(result))
        assert(planParamsMigrate(lastEntry.params).timestamp === baseTimestamp)
        assert(lastEntry.planParamsChangeId === baseId)
        return result
      },
    }),
  }),
})
