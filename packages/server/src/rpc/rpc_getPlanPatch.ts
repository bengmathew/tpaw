// import { block, fGet, letIn, RPC, SomePlanParams } from '@tpaw/common'
// import { RPCContext, RPCFn } from './rpc'
// import { serialTransaction } from '../Utils/PrismaTransaction'
// import _ from 'lodash'
// import { getPlanWithoutParams } from './rpc_getInitialLocalModel'
// import { patchPlanParams } from '../gql/GQLUser/GQLUserPlan/PatchPlanParams'

// export const rpc_getPlanPatch: RPCFn<'getPlanPatch'> = async (
//   { userId, planId, planParamsHistoryEnding: planParamsHistoryEndingClient },
//   ctx,
// ) =>
//   await serialTransaction(
//     async (tx): Promise<RPC.PlanWithParamsPatch | 'unpatchable'> => {
//       const planParamsHistoryEndingServer = await tx.planParamsChange.findMany({
//         where: {
//           userId,
//           planId,
//           timestamp: {
//             gte: new Date(fGet(planParamsHistoryEndingClient[0]).timestamp),
//           },
//         },
//         select: { planParamsChangeId: true, timestamp: true },
//         orderBy: { timestamp: 'asc' },
//       })

//       const cutAfterIndex = letIn(
//         _.zip(
//           planParamsHistoryEndingServer,
//           planParamsHistoryEndingClient,
//         ).findIndex(
//           ([server, client]) =>
//             server?.planParamsChangeId !== client?.id ||
//             // Checking timestamp because changes applied on server will have the
//             // same id, but (possibly) different timestamp.
//             server?.timestamp.getTime() !== client?.timestamp,
//         ),
//         (i) => (i === -1 ? planParamsHistoryEndingServer.length - 1 : i - 1),
//       )
//       // Found nothing in common.
//       if (cutAfterIndex < 0) return 'unpatchable' as const
//       const cutAfter = fGet(planParamsHistoryEndingServer[cutAfterIndex])

//       const dbPlan = await tx.planWithHistory.findUniqueOrThrow({
//         where: { userId_planId: { userId, planId } },
//         include: {
//           paramsChangeHistory: {
//             orderBy: { timestamp: 'desc' },
//             where: { timestamp: { gt: cutAfter.timestamp } },
//           },
//         },
//       })
//       return {
//         ...getPlanWithoutParams(dbPlan),
//         withParams: true,
//         planParamsHistoryPostBase: {
//           reverseHeadIndex: dbPlan.reverseHeadIndex,
//           values: {
//             cutAfterId: cutAfter.planParamsChangeId,
//             append: patchPlanParams(
//               dbPlan.endingParams as SomePlanParams,
//               dbPlan.paramsChangeHistory,
//               'all',
//             ).map((x) => ({
//               id: x.planParamsChangeId,
//               planParams: x.params,
//               change: x.change,
//             })),
//           },
//         },
//       }
//     },
//   )
