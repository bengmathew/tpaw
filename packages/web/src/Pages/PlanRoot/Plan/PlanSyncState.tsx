// import { faTimes } from '@fortawesome/pro-light-svg-icons'
// import {
//     faCheck,
//     faPause,
//     faSpinnerThird,
//     faTriangleExclamation,
// } from '@fortawesome/pro-solid-svg-icons'
// import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
// import { block, noCase } from '@tpaw/common'
// import clsx from 'clsx'
// import { formatDistanceStrict, formatDuration, intervalToDuration } from 'date-fns'
// import _ from 'lodash'
// import React, { useState } from 'react'
// import { useNonPlanParams } from '../PlanRootHelpers/WithNonPlanParams'
// import {
//     SimulationInfoForServerSrc,
//     useSimulationInfo,
// } from '../PlanRootHelpers/WithSimulation'
// import { ServerSyncState } from '../PlanServerImpl/UseServerSyncPlan'


// // TODO: remove showSyncStatus from NonPlanParmas
// // TODO: Delete
// export const PlanSyncState = React.memo(() => {
//   const { nonPlanParams } = useNonPlanParams()
//   const { simulationInfoBySrc } = useSimulationInfo()

//   return simulationInfoBySrc.src !== 'server' ||
//     !nonPlanParams.dev.showSyncStatus ? (
//     <></>
//   ) : (
//     <_Body simulationInfoForServerSrc={simulationInfoBySrc} />
//   )
// })

// const _Body = React.memo(
//   ({
//     simulationInfoForServerSrc,
//   }: {
//     simulationInfoForServerSrc: SimulationInfoForServerSrc
//   }) => {
//     const { syncState } = simulationInfoForServerSrc
//     const [showFailureDetail, setShowFailureDetail] = useState(false)
//     return (
//       <div
//         className="absolute z-70 right-0 bottom-0 bg-pageBG p-2 rounded-tl-lg "
//         style={{ boxShadow: '-4px -4px 14px 4px rgba(0, 0, 0, 0.1)' }}
//       >
//         {block(() => {
//           switch (syncState.type) {
//             case 'synced':
//               return <_Synced state={syncState} />
//             case 'syncing':
//               return (
//                 <_Syncing
//                   state={syncState}
//                   showFailureDetail={showFailureDetail}
//                   setShowFailureDetail={setShowFailureDetail}
//                 />
//               )
//             case 'waitDueToThrottle':
//               return <_WaitDueToThrottle state={syncState} />
//             case 'waitDueToError':
//               return (
//                 <_WaitDueToError
//                   state={syncState}
//                   showFailureDetail={showFailureDetail}
//                   setShowFailureDetail={setShowFailureDetail}
//                 />
//               )
//             default:
//               noCase(syncState)
//           }
//         })}
//       </div>
//     )
//   },
// )

// const _Hide = React.memo(({ className }: { className?: string }) => {
//   const { nonPlanParams, setNonPlanParams } = useNonPlanParams()
//   return (
//     <button
//       className="text-xl font-normal pl-4"
//       onClick={() => {
//         const clone = _.cloneDeep(nonPlanParams)
//         clone.dev.showSyncStatus = false
//         setNonPlanParams(clone)
//       }}
//     >
//       <FontAwesomeIcon icon={faTimes} />
//     </button>
//   )
// })

// const _Synced = React.memo(
//   ({
//     className,
//     state,
//   }: {
//     className?: string
//     state: Extract<ServerSyncState, { type: 'synced' }>
//   }) => {
//     return (
//       <div className={clsx(className, '')}>
//         <div className="flex items-center justify-end">
//           <h2 className="font-bold flex items-center ">
//             <FontAwesomeIcon className="text-lg mr-2" icon={faCheck} />
//             Synced
//           </h2>
//           <_Hide />
//         </div>
//       </div>
//     )
//   },
// )
// const _Syncing = React.memo(
//   ({
//     className,
//     state,
//     showFailureDetail,
//     setShowFailureDetail,
//   }: {
//     className?: string
//     state: Extract<ServerSyncState, { type: 'syncing' }>
//     showFailureDetail: boolean
//     setShowFailureDetail: (x: boolean) => void
//   }) => {
//     const now = useNow()
//     return (
//       <div className={clsx(className)}>
//         <div
//           className="text-sm grid gap-x-2 border-b border-gray-600 mb-2 pb-2"
//           style={{ grid: 'auto/auto auto' }}
//         >
//           <h2 className="text-right">duration:</h2>
//           <h2 className="">{`${((now - state.startTime) / 1000).toFixed(
//             0,
//           )}s`}</h2>
//           <h2 className="text-right">changes:</h2>
//           <h2 className="text-sm ">
//             {_getInputSummaryStr(state.inputSummary)}
//           </h2>
//           <h2 className="text-right">queued:</h2>
//           <h2 className="text-sm ">
//             {state.nextInputSummary
//               ? _getInputSummaryStr(state.nextInputSummary)
//               : '-'}
//           </h2>
//           <h2 className="text-right">failures:</h2>
//           <_Failures
//             failures={state.failures}
//             showDetail={showFailureDetail}
//             setShowDetail={setShowFailureDetail}
//             now={now}
//           />
//         </div>
//         <div className="flex items-center justify-end">
//           <h2 className="font-bold  text-end flex items-center">
//             <FontAwesomeIcon
//               className="fa-spin text-lg mr-2"
//               icon={faSpinnerThird}
//             />
//             Syncing
//           </h2>
//           <_Hide />
//         </div>
//       </div>
//     )
//   },
// )
// const _WaitDueToError = React.memo(
//   ({
//     className,
//     state,
//     showFailureDetail,
//     setShowFailureDetail,
//   }: {
//     className?: string
//     state: Extract<ServerSyncState, { type: 'waitDueToError' }>
//     showFailureDetail: boolean
//     setShowFailureDetail: (x: boolean) => void
//   }) => {
//     const now = useNow()
//     return (
//       <div className={clsx(className)}>
//         <div className="text-sm border-b border-gray-600 mb-2 pb-2">
//           <div className="grid gap-x-2 " style={{ grid: 'auto/auto auto' }}>
//             <h2 className="text-right">eta:</h2>
//             <h2 className="">
//               {state.waitEndTime === 'never'
//                 ? 'never'
//                 : `${((state.waitEndTime - now) / 1000).toFixed(0)}s`}
//             </h2>
//             <h2 className="text-right">changes:</h2>
//             <h2 className="text-sm ">
//               {_getInputSummaryStr(state.queuedSummary)}
//             </h2>
//             <h2 className="text-right">failures:</h2>
//             <_Failures
//               failures={state.failures}
//               showDetail={showFailureDetail}
//               setShowDetail={setShowFailureDetail}
//               now={now}
//             />
//           </div>
//           <div
//             className="flex justify-end pt-2 "
//             onClick={() => state.retryNow()}
//           >
//             <button className="underline">Retry Now</button>
//           </div>
//         </div>
//         <div className="flex items-center justify-end">
//           <h2 className="font-bold text-errorFG text-end flex items-center">
//             <FontAwesomeIcon
//               className="text-lg mr-2"
//               icon={faTriangleExclamation}
//             />
//             Error
//           </h2>
//           <_Hide />
//         </div>
//       </div>
//     )
//   },
// )

// const _WaitDueToThrottle = React.memo(
//   ({
//     className,
//     state,
//   }: {
//     className?: string
//     state: Extract<ServerSyncState, { type: 'waitDueToThrottle' }>
//   }) => {
//     const now = useNow()
//     return (
//       <div className={clsx(className)}>
//         <div className="text-sm border-b border-gray-600 mb-2 pb-2">
//           <div className="grid gap-x-2" style={{ grid: 'auto/auto auto' }}>
//             <h2 className="">eta</h2>
//             <h2 className="">
//               {((state.waitEndTime - now) / 1000).toFixed(0)}s
//             </h2>
//             <h2 className="">queued</h2>
//             <h2 className="">{_getInputSummaryStr(state.queuedSummary)}</h2>
//           </div>
//           <div
//             className="flex justify-end pt-2 "
//             onClick={() => state.runNow()}
//           >
//             <button className="underline">Run Now</button>
//           </div>
//         </div>

//         <div className="flex items-center justify-end">
//           <h2 className="font-bold  text-end flex items-center">
//             <FontAwesomeIcon className="text-lg mr-2" icon={faPause} />
//             Throttle
//           </h2>
//           <_Hide />
//         </div>
//       </div>
//     )
//   },
// )

// const _getInputSummaryStr = (
//   inputSummary: Extract<ServerSyncState, { type: 'syncing' }>['inputSummary'],
// ) =>
//   inputSummary
//     .map((x) =>
//       x.type === 'addItems'
//         ? `add ${x.addCount}`
//         : x.type === 'moveHead'
//         ? 'undo/redo'
//         : x.type === 'cutBranch'
//         ? `cut ${x.cutCount}`
//         : noCase(x),
//     )
//     .join(', ')

// const _Failures = React.memo(
//   ({
//     className,
//     failures,
//     showDetail,
//     setShowDetail,
//     now,
//   }: {
//     className?: string
//     failures: Extract<ServerSyncState, { type: 'syncing' }>['failures']
//     showDetail: boolean
//     setShowDetail: (show: boolean) => void
//     now: number
//   }) => {
//     if (failures.length === 0) {
//       return (
//         <div className={clsx(className)}>
//           <h2 className="">-</h2>
//         </div>
//       )
//     }

//     if (!showDetail) {
//       return (
//         <button
//           className={clsx(className, 'text-start')}
//           onClick={() => setShowDetail(true)}
//         >
//           <h2 className="">
//             {`${failures.length} (last: ${
//               failures[failures.length - 1].reason
//             })`}
//           </h2>
//         </button>
//       )
//     } else {
//       return (
//         <button
//           className={clsx(
//             className,
//             'text-start block max-h-[calc(100vh-150px)] overflow-y-scroll',
//           )}
//           onClick={() => setShowDetail(false)}
//         >
//           {failures
//             .slice()
//             .reverse()
//             .map((failure, i) => (
//               <div
//                 key={i}
//                 className="border-b border-t border-gray-300 text-xs"
//               >
//                 <h2 className="">{`${formatDistanceStrict(
//                   failure.timing.start,
//                   now,
//                 )} ago`}</h2>
//                 <h2 className="">
//                   {`for ${formatDuration(intervalToDuration(failure.timing))}`}
//                 </h2>
//                 <h2 className="">{failure.reason}</h2>
//               </div>
//             ))}
//         </button>
//       )
//     }
//   },
// )

// const useNow = () => {
//   const [now, setNow] = useState(Date.now())
//   React.useEffect(() => {
//     const interval = window.setInterval(() => {
//       setNow(Date.now())
//     }, 500)
//     return () => window.clearInterval(interval)
//   }, [])
//   return now
// }
