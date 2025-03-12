// import {
//   ApplyPlanUpdateAction,
//   assert,
//   block,
//   fGet,
//   letIn,
//   noCase,
//   planParamsMigrate,
//   ReverseHeadIndex,
//   RPC,
//   SomePlanParams,
// } from '@tpaw/common'
// import _ from 'lodash'
// import {
//   PrismaTransaction,
//   serialTransaction,
// } from '../Utils/PrismaTransaction'
// import { patchPlanParams } from '../gql/GQLUser/GQLUserPlan/PatchPlanParams'
// import { RPCError, RPCFn } from './rpc'

// export const rpc_syncAndGetPatches: RPCFn<'syncAndGetPatches'> = async (
//   { syncActions, ianaTimezoneName },
//   ctx,
// ) => {
//   const updatePlanActionsByPlanId: Map<string, RPC.PlanAction[][]> = new Map()

//   syncActions.forEach((x) => {
//     switch (x.type) {
//       case 'updatePlan':
//         if (x.userId !== ctx.userId) throw new RPCError('unauthorized')
//         updatePlanActionsByPlanId.set(x.planId, [
//           ...(updatePlanActionsByPlanId.get(x.planId) ?? []),
//           x.planUpdateAction,
//         ])
//         break
//       default:
//         noCase(x.type)
//     }
//   })

//   const planResults = await Promise.all(
//     Array.from(updatePlanActionsByPlanId.entries()).map(
//       async ([planId, updateActions]) => ({
//         ...(await updatePlan(
//           fGet(ctx.userId),
//           planId,
//           updateActions,
//           ianaTimezoneName,
//         )),
//         planId,
//       }),
//     ),
//   )
//   return { plans: planResults }
// }

// namespace SyncPlan {
//   const sync = async (
//     {
//       userId,
//       planId,
//       patchArgsIfNotDeleted,
//       planActions,
//     }: RPC.Args<'syncAndGetPatches'>['plans'][number],
//     ianaTimezoneName: string,
//   ) => {
//     const transactions = _splitIntoTransactions(planActions)
//     transactions.forEach((x) => {
//       switch (x.type) {
//         case 'create':
//           // TODO:
//           break
//         case 'reset':
//           // TODO:
//           break
//         case 'delete':
//           // TODO:
//           break
//         case 'copy':
//           // TODO:
//           break
//         case 'update':
//           // TODO:
//           break
          
//       }
//     })
//   }
//   }

//   type Transaction =
//     | { type: 'create'; action: Extract<RPC.PlanAction, { type: 'create' }> }
//     | { type: 'reset' }
//     | { type: 'delete' }
//     | { type: 'copy'; action: Extract<RPC.PlanAction, { type: 'copy' }> }
//     | {
//         type: 'update'
//         actions: (
//           | Extract<RPC.PlanAction, { type: 'setLabel' }>
//           | Extract<RPC.PlanAction, { type: 'planParamChange' }>
//           | Extract<RPC.PlanAction, { type: 'setReverseHeadIndex' }>
//         )[]
//       }

//   const _splitIntoTransactions = (planActions: RPC.PlanAction[]) => {
//     const transactions: Transaction[] = []
//     const _popLastIfUpdate = () => {
//       const last = _.last(transactions)
//       if (last?.type === 'update') {
//         transactions.pop()
//         return last
//       }
//       return null
//     }

//     for (const action of planActions) {
//       switch (action.type) {
//         case 'create':
//           transactions.push({ type: 'create', action })
//           break
//         case 'delete':
//         case 'reset':
//           _popLastIfUpdate()
//           transactions.push({ type: action.type })
//           break
//         case 'copy':
//           transactions.push({ type: 'copy', action })
//           break
//         case 'setLabel':
//         case 'planParamChange':
//         case 'setReverseHeadIndex':
//           const transaction = _popLastIfUpdate() ?? {
//             type: 'update',
//             actions: [],
//           }
//           transaction.actions.push(action)
//           transactions.push(transaction)
//           break
//         default:
//           noCase(action)
//       }
//     }
//   }

//   const updatePlan = async (
//     userId: string,
//     planId: string,
//     { actions }: Extract<Transaction, { type: 'update' }>,
//     ianaTimezoneName: string,
//   ) =>
//     await serialTransaction(async (tx) => {
//       const workingPlan = await _getWorkingPlan(
//         tx,
//         userId,
//         planId,
//         actions,
//       )
//       const { workingPlan: newWorkingPlan, failedCount } =
//         await _applyToWorkingPlan(workingPlan, actions, ianaTimezoneName)
//       await _commit(tx, userId, planId, workingPlan, newWorkingPlan)

//       return {
//         failureInfo: {
//           count: failedCount,
//           percentage: failedCount / actions.length,
//         },
//       }
//     })

//   const REVERSE_HEAD_INDEX_GUESS = 5
//   const _getReverseHeadId = async (
//     tx: PrismaTransaction,
//     userId: string,
//     planId: string,
//   ) => {
//     const helper = async (reverseIndexGuess: number) => {
//       const { reverseHeadIndex, paramsChangeHistory } =
//         await tx.planWithHistory.findUniqueOrThrow({
//           where: { userId_planId: { userId, planId } },
//           include: {
//             paramsChangeHistory: {
//               orderBy: { timestamp: 'desc' },
//               take: reverseIndexGuess + 1,
//               select: {
//                 timestamp: true,
//                 planParamsChangeId: true,
//               },
//             },
//           },
//         })
//       return {
//         reverseHeadIndex,
//         changeId:
//           paramsChangeHistory.length > reverseHeadIndex
//             ? fGet(
//                 paramsChangeHistory[
//                   ReverseHeadIndex.toHeadIndex(
//                     reverseHeadIndex,
//                     paramsChangeHistory.length,
//                   )
//                 ],
//               ).planParamsChangeId
//             : null,
//       }
//     }
//     const info = await helper(REVERSE_HEAD_INDEX_GUESS)
//     return info.changeId ?? fGet((await helper(info.reverseHeadIndex)).changeId)
//   }

//   type WorkingPlan = Awaited<ReturnType<typeof _getWorkingPlan>>
//   const _getWorkingPlan = async (
//     tx: PrismaTransaction,
//     userId: string,
//     planId: string,
//     updateActions: RPC.PlanUpdateAction[],
//   ) => {
//     const selectIds = [
//       ..._.compact(
//         updateActions
//           .map((x) => (x.type === 'setReverseHeadIndex' ? x.targetId : null))
//           .filter((x) => x !== null),
//       ),
//       await _getReverseHeadId(tx, userId, planId),
//     ]

//     const workingPlanStartTimestamp = fGet(
//       _.first(
//         (
//           await tx.planWithHistory.findUniqueOrThrow({
//             where: { userId_planId: { userId, planId } },
//             include: {
//               paramsChangeHistory: {
//                 where: { planParamsChangeId: { in: selectIds } },
//                 orderBy: { timestamp: 'asc' },
//                 select: {
//                   timestamp: true,
//                   planParamsChangeId: true,
//                 },
//               },
//             },
//           })
//         ).paramsChangeHistory,
//       ),
//     ).timestamp

//     const plan = await tx.planWithHistory.findUniqueOrThrow({
//       where: { userId_planId: { userId, planId } },
//       include: {
//         paramsChangeHistory: {
//           orderBy: { timestamp: 'desc' },
//           where: {
//             timestamp: {
//               gte: workingPlanStartTimestamp,
//             },
//           },
//         },
//       },
//     })

//     return {
//       planParamsHistory: patchPlanParams(
//         plan.endingParams as SomePlanParams,
//         plan.paramsChangeHistory,
//         'all',
//       ).map((x) => ({
//         id: x.planParamsChangeId,
//         change: x.change,
//         planParamsUnmigrated: x.params,
//         planParams: planParamsMigrate(x.params),
//       })),
//       reverseHeadIndex: plan.reverseHeadIndex,
//     }
//   }

//   const _applyToWorkingPlan = async (
//     workingPlan: ApplyPlanUpdateAction.WorkingPlan,
//     updateActions: RPC.PlanUpdateAction[],
//     ianaTimezoneName: string,
//   ) => {
//     let failedCount = 0
//     class StopError extends Error {}

//     try {
//       updateActions.forEach((updateAction, i) => {
//         switch (updateAction.type) {
//           case 'planParamChange': {
//             const now = Date.now()
//             const result = ApplyPlanUpdateAction.planParamsChangeAction(
//               workingPlan,
//               updateAction,
//               now,
//               now,
//               ianaTimezoneName,
//               {
//                 allowMerge: i > 0, // Don't merge across syncs.
//                 allowNoOp: false,
//                 allowFailure: true,
//               },
//             )
//             assert(result !== 'noOp')
//             if (result === 'failed') {
//               failedCount++
//             } else {
//               workingPlan = result
//             }
//             break
//           }
//           case 'setReverseHeadIndex':
//             const result = ApplyPlanUpdateAction.setReverseHeadIndex(
//               workingPlan,
//               updateAction,
//               { allowFailure: true },
//             )
//             if (result === 'failed') {
//               failedCount += updateActions.length - i
//               throw new StopError()
//             } else {
//               workingPlan = result
//             }
//             break
//           default:
//             noCase(updateAction)
//         }
//       })
//     } catch (e) {
//       if (!(e instanceof StopError)) {
//         throw e
//       }
//     }
//     {
//       const timestamps = workingPlan.planParamsHistory.map(
//         (x) => x.planParams.timestamp,
//       )
//       const sortedUniqueTimestamps = _.uniq([...timestamps].sort())
//       assert(_.isEqual(timestamps, sortedUniqueTimestamps))
//     }
//     return { workingPlan, failedCount }
//   }

//   const _commit = async (
//     tx: PrismaTransaction,
//     userId: string,
//     planId: string,
//     workingPlan: WorkingPlan,
//     newWorkingPlan: ApplyPlanUpdateAction.WorkingPlan,
//   ) => {
//     const { cutAfter, add } = block(() => {
//       const cutAfterIndex = letIn(
//         _.zip(
//           workingPlan.planParamsHistory,
//           newWorkingPlan.planParamsHistory,
//         ).findIndex(([a, b]) => a?.id !== b?.id),
//         (i) => (i === -1 ? workingPlan.planParamsHistory.length - 1 : i - 1),
//       )
//       // There should be at least one common history item.
//       assert(cutAfterIndex >= 0)
//       return {
//         cutAfter: letIn(
//           fGet(workingPlan.planParamsHistory[cutAfterIndex]),
//           (x) => ({
//             planParamsUnmigrated: x.planParamsUnmigrated,
//             timestamp: x.planParams.timestamp,
//           }),
//         ),
//         // slice() returns empty array if index is past the end.
//         add: newWorkingPlan.planParamsHistory.slice(cutAfterIndex + 1),
//       }
//     })

//     const now = new Date()
//     await tx.planWithHistory.update({
//       where: { userId_planId: { userId, planId } },
//       data: {
//         lastSyncAt: now,
//         sortTime: now,
//         paramsChangeHistory: {
//           deleteMany: {
//             timestamp: { gt: new Date(cutAfter.timestamp) },
//           },
//           createMany: {
//             data: patchPlanParams.generate(
//               {
//                 type: 'forAdd',
//                 params: cutAfter.planParamsUnmigrated,
//                 timestamp: cutAfter.timestamp,
//               },
//               add.map((x) => ({
//                 id: x.id,
//                 params: x.planParams,
//                 change: x.change,
//               })),
//             ),
//           },
//         },
//         endingParams: _.last(add)?.planParams ?? cutAfter.planParamsUnmigrated,
//         reverseHeadIndex: newWorkingPlan.reverseHeadIndex,
//       },
//     })
//   }
// }
