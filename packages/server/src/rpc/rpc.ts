// import asyncHandler from 'express-async-handler'
// import * as z from 'zod'
// import { block, noCase, RPC } from '@tpaw/common'
// import { rpc_getInitialLocalModel } from './rpc_getInitialLocalModel.js'
// import { Clients } from '../Clients.js'

// // TODO: sentry

// export class RPCError extends Error {
//   constructor(readonly reason: 'unauthorized') {
//     super()
//   }
// }

// export type RPCContext = {
//   userId: string | null
// }

// export type RPCFn<T extends RPC.MethodName> = (
//   args: RPC.Args<T>,
//   ctx: RPCContext,
// ) => Promise<RPC.Result<T>>

// export const rpcHandler = asyncHandler(async (req, res) => {
//   try {
//     const mainArgsCheck = RPC.mainArgs.safeParse(req.body)
//     if (!mainArgsCheck.success) {
//       res.status(400).send() // Bad request
//       return
//     }
//     const { method, args } = mainArgsCheck.data
//     const userId = await block<Promise<string | null>>(async () => {
//       const idToken = (req.headers.authorization || '')
//         .split(', ')
//         .filter((x) => x.startsWith('Bearer'))
//         .map((x) => x.substring('Bearer '.length))[0]

//       if (!idToken) return null
//       const decodedToken = await Clients.firebaseAuth.verifyIdToken(idToken)
//       return decodedToken.uid
//     })

//     res.send(
//       JSON.stringify(
//         // eslint-disable-next-line @typescript-eslint/no-unsafe-argument, @typescript-eslint/no-explicit-any
//         await impl[`rpc_${method}`](args as any, { userId }),
//       ),
//     )
//   } catch (e) {
//     if (e instanceof RPCError) {
//       switch (e.reason) {
//         case 'unauthorized':
//           res.status(401).send() // Unauthorized
//           break
//         default:
//           noCase(e.reason)
//       }
//     }
//     throw e
//   }
// })

// const impl: {
//   [K in RPC.MethodName as `rpc_${K}`]: RPCFn<K>
// } = {
//   rpc_getInitialLocalModel,
//   rpc_getPlanParamsHistoryPreBase,
//   rpc_syncAndGetPatches,
//   rpc_getPlanPatch,
// }
