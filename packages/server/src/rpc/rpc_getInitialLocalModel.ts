// import { PlanParamsChange, PlanWithHistory } from '@prisma/client'
// import {
//   assert,
//   block,
//   PlanParamsChangeAction,
//   PlanParamsHistoryStoreFns,
//   ReverseHeadIndex,
//   RPC,
//   SomePlanParams,
// } from '@tpaw/common'
// import { Operation } from 'fast-json-patch'
// import { serialTransaction } from '../Utils/PrismaTransaction.js'
// import { RPCError, RPCFn } from './rpc.js'

// export const rpc_getInitialLocalModel: RPCFn<'getInitialLocalModel'> = async (
//   { userId, minPlanParamsPostBaseSize, additionalPlanId },
//   ctx,
// ) => {
//   if (ctx.userId !== userId) throw new RPCError('unauthorized')
//   const user = await serialTransaction((tx) =>
//     tx.user.findUniqueOrThrow({
//       where: { id: userId },
//       include: {
//         planWithHistory: true,
//       },
//     }),
//   )

//   return {
//     user: {
//       userId,
//     },
//     plans: await Promise.all(
//       user.planWithHistory.map((x) =>
//         x.isMain || x.planId === additionalPlanId
//           ? getPlanWithParams(userId, x.planId, minPlanParamsPostBaseSize)
//           : getPlanWithoutParams(x),
//       ),
//     ),
//   }
// }

// export const getPlanWithoutParams = (
//   dbPlan: PlanWithHistory,
// ): RPC.PlanWithoutParams => ({
//   planId: dbPlan.planId,
//   isMain: dbPlan.isMain,
//   slug: dbPlan.slug,
//   label: dbPlan.label,
//   addedToServerAt: dbPlan.addedToServerAt.getTime(),
//   sortTime: dbPlan.sortTime.getTime(),
//   lastSyncAt: dbPlan.lastSyncAt.getTime(),
//   resetCount: dbPlan.resetCount,
//   endingPlanParams: dbPlan.endingParams as SomePlanParams,
//   reverseHeadIndex: dbPlan.reverseHeadIndex,
// })

// export const getPlanWithParams = async (
//   userId: string,
//   planId: string,
//   minPlanParamsPostBaseSize: number,
// ): Promise<RPC.PlanWithParams> =>
//   serialTransaction(async (tx): Promise<RPC.PlanWithParams> => {
//     const dbPlan = await tx.planWithHistory.findUniqueOrThrow({
//       where: { userId_planId: { userId, planId } },
//       include: {
//         paramsChangeHistory: {
//           orderBy: { timestamp: 'desc' },
//         },
//       },
//     })
//     assert(minPlanParamsPostBaseSize > dbPlan.reverseHeadIndex)
//     const dbPlanParamsHistory = dbPlan.paramsChangeHistory.map(
//       planParamsHistoryItemToCommonStored,
//     )

//     const indexOfBase = block(() => {
//       const planParamsHistory = PlanParamsHistoryStoreFns.fromStored(
//         dbPlan.endingParams as SomePlanParams,
//         dbPlanParamsHistory,
//       )
//       const requestedBase = ReverseHeadIndex.fGet(
//         minPlanParamsPostBaseSize - 1,
//         planParamsHistory,
//       )
//       const basePortfolioBalance =
//         requestedBase.planParams.wealth.portfolioBalance

//       if (!basePortfolioBalance.isDatedPlan || basePortfolioBalance.updatedHere)
//         return ReverseHeadIndex.toHeadIndex(
//           minPlanParamsPostBaseSize - 1,
//           planParamsHistory.length,
//         )

//       const index = planParamsHistory.findIndex(
//         (x) => x.id === basePortfolioBalance.updatedAtId,
//       )
//       assert(index !== -1)
//       return index
//     })

//     const planParamsHistoryPostBase = dbPlanParamsHistory.slice(indexOfBase)

//     return { ...getPlanWithoutParams(dbPlan), planParamsHistoryPostBase }
//   })

// export const planParamsHistoryItemToCommonStored = (
//   item: PlanParamsChange,
// ): PlanParamsHistoryStoreFns.ItemStored => ({
//   id: item.planParamsChangeId,
//   timestamp: item.timestamp.getTime(),
//   change: item.change as PlanParamsChangeAction,
//   reverseDiff: item.reverseDiff as unknown as Operation[],
// })
