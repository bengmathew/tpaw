import {
  API,
  PlanParams,
  PlanParamsChangeActionCurrent,
  SomePlanParams,
  assert,
  fGet,
  letIn,
  planParamsMigrate,
} from '@tpaw/common'
import _ from 'lodash'
import { Clients } from '../../../../Clients.js'
import { PrismaTransaction } from '../../../../Utils/PrismaTransaction.js'
import { concurrentChangeError } from '../../../../impl/Common/ConcurrentChangeError.js'
import { PothosPlanAndUserResult } from '../../../GQLCommon/GQLPlanAndUserResult.js'
import { builder } from '../../../builder.js'
import { patchPlanParams } from '../PatchPlanParams.js'

const Input = builder.inputType('UserPlanSyncInput', {
  fields: (t) => ({
    userId: t.string(),
    planId: t.string(),
    lastSyncAt: t.float(),
    cutAfterId: t.string(),
    add: t.field({
      type: [
        builder.inputType('UserPlanSyncAddInput', {
          fields: (t) => ({
            id: t.string(),
            params: t.string(),
            change: t.string(),
          }),
        }),
      ],
    }),
    reverseHeadIndex: t.int(),
  }),
})

builder.mutationField('userPlanSync', (t) =>
  t.field({
    type: builder.unionType('UserPlanSyncResult', {
      types: ['ConcurrentChangeError', PothosPlanAndUserResult],
      resolveType: (value) => value.type,
    }),
    authScopes: (_, args, ctx) => ctx.user?.id === args.input.userId,
    args: { input: t.arg({ type: Input }) },
    resolve: async (__, { input }) => {
      const { userId, planId, lastSyncAt, cutAfterId, add, reverseHeadIndex } =
        API.UserPlanSync.check(input).force()

      return await Clients.prisma.$transaction(async (tx) => {
        const startingPlan = await tx.planWithHistory.findUniqueOrThrow({
          where: { userId_planId: { userId, planId } },
        })
        if (startingPlan.lastSyncAt.getTime() !== lastSyncAt)
          return concurrentChangeError

        await userPlanSync(
          tx,
          userId,
          planId,
          cutAfterId,
          add,
          reverseHeadIndex,
        )

        return { type: 'PlanAndUserResult' as const, planId, userId }
      })
    },
  }),
)

export const userPlanSync = async (
  tx: PrismaTransaction,
  userId: string,
  planId: string,
  cutAfterId: string,
  add: {
    id: string
    params: PlanParams
    change: PlanParamsChangeActionCurrent
  }[],
  reverseHeadIndex: number,
) => {
  const cutAfterHistoryItem = await tx.planParamsChange.findUniqueOrThrow({
    where: {
      userId_planId_planParamsChangeId: {
        userId,
        planId,
        planParamsChangeId: cutAfterId,
      },
    },
  })

  if (add.length > 0) {
    assert(
      fGet(add[0]).params.timestamp > cutAfterHistoryItem.timestamp.getTime(),
    )
  }

  const paramsAtCut = letIn(
    await tx.planWithHistory.findUniqueOrThrow({
      where: { userId_planId: { userId, planId } },
      include: {
        paramsChangeHistory: {
          where: {
            timestamp: { gte: new Date(cutAfterHistoryItem.timestamp) },
          },
          orderBy: { timestamp: 'desc' },
        },
      },
    }),
    (x) => {
      return fGet(
        patchPlanParams.forSingle(
          x.endingParams as SomePlanParams,
          x.paramsChangeHistory,
        ),
      ).params
    },
  )
  assert(
    planParamsMigrate(paramsAtCut).timestamp ===
      cutAfterHistoryItem.timestamp.getTime(),
  )

  const now = new Date()
  await tx.planWithHistory.update({
    where: { userId_planId: { userId, planId } },
    data: {
      lastSyncAt: now,
      sortTime: now,
      paramsChangeHistory: {
        deleteMany: {
          timestamp: { gt: new Date(cutAfterHistoryItem.timestamp) },
        },
        createMany: {
          data: patchPlanParams.generate(
            {
              type: 'forAdd',
              params: paramsAtCut,
              timestamp: cutAfterHistoryItem.timestamp.getTime(),
            },
            add,
          ),
        },
      },
      endingParams: _.last(add)?.params ?? paramsAtCut,
      reverseHeadIndex,
    },
  })
}
