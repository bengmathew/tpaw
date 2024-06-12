import * as Sentry from '@sentry/nextjs'
import { assert, assertFalse, block, fGet, noCase } from '@tpaw/common'
import _ from 'lodash'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useMutation } from 'react-relay'
import { graphql } from 'relay-runtime'
import { AppError } from '../../App/AppError'
import { useSetGlobalError } from '../../App/GlobalErrorBoundary'
import { WorkingPlanInfo } from '../PlanRootHelpers/UseWorkingPlan'
import {
  UseServerSyncPlanMutation,
  UserPlanSyncAddInput,
} from './__generated__/UseServerSyncPlanMutation.graphql'

export const SERVER_SYNC_PLAN_THROTTLE_WAIT_TIME = 5 * 1000
export const SERVER_SYNC_PLAN_ERROR_WAIT_TIME = 20 * 1000
const _TIMEOUT = 60 * 1000

type _ServerPlan = {
  planId: string
  lastSyncAt: number
  planParamsPostBaseIds: readonly string[]
  reverseHeadIndex: number
}

type _Failure = {
  reason:
    | 'serverDownForMaintenance'
    | 'serverDownForUpdate'
    | 'timeout'
    | 'other'
  timing: { start: number; end: number }
}

type _State =
  | { type: 'synced' }
  | {
      type: 'syncing'
      startTime: number
      failures: _Failure[]
      inputSummary: _Input['changeSummary']
      getInputSummaryAfterSync: (
        workingPlan: WorkingPlanInfo['workingPlan'],
      ) => _Input['changeSummary'] | null
      dispose: () => void
    }
  | {
      type: 'waitDueToError'
      waitEndTime: number | 'never'
      failures: _Failure[]
      dispose: () => void
    }
  | { type: 'waitDueToThrottle'; waitEndTime: number; dispose: () => void }

export type ServerSyncState =
  | { type: 'synced' }
  | {
      type: 'syncing'
      startTime: number
      failures: _Failure[]
      inputSummary: _Input['changeSummary']
      nextInputSummary: _Input['changeSummary'] | null
    }
  | {
      type: 'waitDueToError'
      waitEndTime: number | 'never'
      failures: _Failure[]
      queuedSummary: _Input['changeSummary']
      retryNow: () => void
    }
  | {
      type: 'waitDueToThrottle'
      waitEndTime: number
      queuedSummary: _Input['changeSummary']
      runNow: () => void
    }

export const useServerSyncPlan = (
  userId: string,
  serverPlan: _ServerPlan,
  workingPlan: WorkingPlanInfo['workingPlan'],
  setServerPlan: (x: _ServerPlan) => void,
): ServerSyncState => {
  const { setGlobalError } = useSetGlobalError()
  const [state, setStateDirect] = useState<_State>({ type: 'synced' })

  const historyForDebugRef = useRef([] as string[])
  const debug = (log: string) => {
    historyForDebugRef.current.push(log)
    historyForDebugRef.current = historyForDebugRef.current.slice(-20)
  }
  const debugRef = useRef(debug)
  debugRef.current = debug

  const setStateDebug = (
    src: string | null,
    action: string,
    targetState: _State,
  ) => {
    setStateDirect((prev) => {
      debugRef.current(
        JSON.stringify(
          {
            src,
            action,
            currState: prev.type,
            targetState: targetState.type,
          },
          null,
          2,
        ),
      )
      if (state.type !== prev.type) {
        const message = `Double setState called:\n${historyForDebugRef.current.join('\n\n')}`
        console.dir(message)
        Sentry.captureMessage(message)
        assertFalse()
      }
      return targetState
    })
  }

  // ---- INPUT ----
  // Use custom memoization because useMemo() is not guaranteed and we need
  // the guarantee for correctness because of dependency array.
  const input = _getInputMemoized(serverPlan, workingPlan)

  // ---- TO SYNCED STATE ----
  const toSyncedState = () => {
    setStateDebug(null, 'toSyncedState', { type: 'synced' })
  }
  const toSyncedStateRef = useRef(toSyncedState)
  toSyncedStateRef.current = toSyncedState

  // ---- TO SYNCING STATE ----
  const [commitSync] = useMutation<UseServerSyncPlanMutation>(graphql`
    mutation UseServerSyncPlanMutation($input: UserPlanSyncInput!) {
      userPlanSync(input: $input) {
        __typename
        ... on PlanAndUserResult {
          plan {
            id
            lastSyncAt
            ...PlanWithoutParamsFragment
          }
        }
        ... on ConcurrentChangeError {
          _
        }
      }
    }
  `)

  const toSyncingState = (
    src: string,
    {
      input,
      failures,
    }: {
      input: _Input
      failures: _Failure[]
    },
  ) => {
    if (state.type === 'syncing') {
      Sentry.captureException(
        new Error(
          `toSyncingState called while already syncing.\nsrc:${src}\n${JSON.stringify(historyForDebugRef.current, null, 2)}`,
        ),
      )
      assertFalse()
    }
    const { cutAfterId, add, reverseHeadIndex, newParamsHistoryPostBaseIds } =
      input
    const startTime = Date.now()
    const timeout = window.setTimeout(() => {
      disposeCall()
      toErrorStateRef.current({
        reason: 'timeout',
        timing: { start: startTime, end: Date.now() },
      })
    }, _TIMEOUT)
    const disposeTimeout = () => window.clearTimeout(timeout)
    const { dispose: disposeCall } = commitSync({
      variables: {
        input: {
          userId,
          planId: serverPlan.planId,
          lastSyncAt: serverPlan.lastSyncAt,
          cutAfterId,
          add,
          reverseHeadIndex,
        },
      },

      // Note, throwing error here does not trigger onError.
      onCompleted: ({ userPlanSync }) => {
        disposeTimeout()
        switch (userPlanSync.__typename) {
          case 'ConcurrentChangeError':
            Sentry.captureMessage(
              `ConcurrentChangeError\n: ${JSON.stringify(historyForDebugRef.current, null, 2)}`,
            )
            setGlobalError(new AppError('concurrentChange'))
            break
          case 'PlanAndUserResult':
            const newServerPlan: _ServerPlan = {
              planId: serverPlan.planId,
              lastSyncAt: userPlanSync.plan.lastSyncAt,
              planParamsPostBaseIds: newParamsHistoryPostBaseIds,
              reverseHeadIndex,
            }
            setServerPlan(newServerPlan)
            toThrottleStateRef.current()
            break
          case '%other':
            setGlobalError(new Error('Did not expect to get here.'))
            break
          default:
            noCase(userPlanSync)
        }
      },
      onError: (e) => {
        disposeTimeout()
        const reason = block<_Failure['reason'] | null>(() => {
          if (!(e instanceof AppError)) {
            Sentry.captureException(e)
            return 'other'
          }
          switch (e.code) {
            case 'concurrentChange':
              // FetchGQL will not throw this. It will show up as
              // 'ConcurrentChangeError' in onCompleted().
              assertFalse()
            case 'clientNeedsUpdate':
            case '413':
              if (e.code === '413') {
                Sentry.captureMessage(
                  `413 in UseServerSyncPlan.\naddCount: ${add.length}`,
                )
              }
              setGlobalError(e)
              return null
            case '404':
            case 'networkError':
            case 'serverError':
              Sentry.captureException(e)
              return 'other'
            case 'serverDownForMaintenance':
            case 'serverDownForUpdate':
              return e.code
            default:
              noCase(e.code)
          }
        })
        if (!reason) return
        toErrorStateRef.current({
          reason,
          timing: { start: startTime, end: Date.now() },
        })
      },
    })
    setStateDebug(src, 'toSyncingState', {
      type: 'syncing',
      startTime,
      failures,
      inputSummary: input.changeSummary,
      getInputSummaryAfterSync: (workingPlan) =>
        _getInputMemoized(
          {
            planParamsPostBaseIds: newParamsHistoryPostBaseIds,
            reverseHeadIndex,
          },
          workingPlan,
        )?.changeSummary ?? null,
      dispose: () => {
        disposeTimeout()
        disposeCall()
      },
    })
  }
  const toSyncingStateRef = useRef(toSyncingState)
  toSyncingStateRef.current = toSyncingState

  // ---- TO THROTTLE STATE ----
  const handleClearThrottle = () => {
    assert(input)
    if (state.type !== 'waitDueToThrottle') {
      throw new Error(
        `Expected state.type to be 'waitDueToThrottle', but was ${state.type}.`,
      )
    }
    state.dispose() // Safe even if timeout has already fired.
    toSyncingStateRef.current('handleClearThrottle', { input, failures: [] })
  }
  const handleClearThrottleRef = useRef(handleClearThrottle)
  handleClearThrottleRef.current = handleClearThrottle
  const toThrottleState = () => {
    if (state.type !== 'syncing') {
      Sentry.captureException(
        new Error(
          `Expected state.type to be 'syncing', but was ${state.type}.\n${JSON.stringify(historyForDebugRef.current, null, 2)}`,
        ),
      )
      assertFalse()
    }
    const timeout = window.setTimeout(
      () => handleClearThrottleRef.current(),
      SERVER_SYNC_PLAN_THROTTLE_WAIT_TIME,
    )
    setStateDebug(null, 'toThrottleState', {
      type: 'waitDueToThrottle',
      waitEndTime: Date.now() + SERVER_SYNC_PLAN_THROTTLE_WAIT_TIME,
      dispose: () => window.clearTimeout(timeout),
    })
  }
  const toThrottleStateRef = useRef(toThrottleState)
  toThrottleStateRef.current = toThrottleState

  // ---- TO ERROR STATE ----
  const handleClearError = () => {
    assert(input)
    if (state.type !== 'waitDueToError') {
      Sentry.captureException(
        new Error(
          `Expected state.type to be 'waitDueToError', but was ${state.type}.\n${JSON.stringify(historyForDebugRef.current, null, 2)}`,
        ),
      )
      assertFalse()
    }
    state.dispose() // Safe even if timeout has already fired.
    toSyncingStateRef.current('handleClearError', {
      input,
      failures: state.failures,
    })
  }
  const handleClearErrorRef = useRef(handleClearError)
  handleClearErrorRef.current = handleClearError

  const toErrorState = (failure: _Failure) => {
    if (state.type !== 'syncing') {
      Sentry.captureException(
        new Error(
          `Expected state.type to be 'syncing', but was ${state.type}.\n${JSON.stringify(historyForDebugRef.current, null, 2)}`,
        ),
      )
      assertFalse()
    }
    const waitEndTime = block(() => {
      switch (failure.reason) {
        // Don't retry on timeout, the server might have handled the request.
        // Sending it again might cause a concurrentChange error, which will
        // be confusing to the user. Better to force them to reload the page.
        case 'timeout':
          return 'never' as const
        case 'serverDownForMaintenance':
        case 'serverDownForUpdate':
        case 'other':
          return Date.now() + SERVER_SYNC_PLAN_ERROR_WAIT_TIME
        default:
          noCase(failure.reason)
      }
    })

    setStateDebug(null, 'toErrorState', {
      type: 'waitDueToError',
      waitEndTime: waitEndTime,
      failures: [...state.failures, failure],
      dispose:
        waitEndTime === 'never'
          ? () => {}
          : block(() => {
              const timeout = window.setTimeout(
                () => handleClearErrorRef.current(),
                SERVER_SYNC_PLAN_ERROR_WAIT_TIME,
              )
              return () => window.clearTimeout(timeout)
            }),
    })
  }
  const toErrorStateRef = useRef(toErrorState)
  toErrorStateRef.current = toErrorState

  // ---- RESPOND TO INPUT CHANGES  ----
  const debugLastChangeRef = useRef(
    null as null | {
      state: typeof state
      input: typeof input
      trace: 'start' | 'end'
    },
  )
  const isEffectRunningRef = useRef(false)
  isEffectRunningRef.current = false
  const [effectTriggerDelay, setEffectTriggerDelay] = useState(0)
  useEffect(() => {
    if (isEffectRunningRef.current) {
      setEffectTriggerDelay((prev) => prev + 1)
      return
    }
    isEffectRunningRef.current = true
    const whatChanged = block(() => {
      const last = debugLastChangeRef.current
      const inputChanged = last ? last.input !== input : true
      const stateChanged = last ? last.state !== state : true
      const whatChanged =
        inputChanged && stateChanged
          ? 'both'
          : inputChanged
            ? 'input'
            : stateChanged
              ? 'state'
              : 'neither'

      const info = {
        whatChanged,
        state,
        ...(inputChanged
          ? { inputDeepChanged: !_.isEqual(last?.input, input) }
          : {}),
        inputIsNull: input === null,
        ...(last
          ? {
              lastInputIsNull: last.input === null,
              lastState: last.state,
              lastTrace: last.trace,
            }
          : { last: null }),
      }
      debugRef.current(
        `checkingInputAndState\n${JSON.stringify(info, null, 2)}`,
      )
      return whatChanged
    })
    debugLastChangeRef.current = { state, input, trace: 'start' }
    if (input) {
      switch (state.type) {
        case 'synced':
          toSyncingStateRef.current(`inputChangeEffect-${whatChanged}`, {
            input,
            failures: [],
          })
          break
        case 'syncing':
        case 'waitDueToError':
        case 'waitDueToThrottle':
          break
        default:
          noCase(state)
      }
    } else {
      // Not sure if any case except 'synced' in possible, but handle it anyway.
      // Actually this is possible. Eg. undo and redo that cancels the undo.
      switch (state.type) {
        case 'synced':
        // Will let sync finish. input will probably change to non-null due to
        // server state being updated at sync end.
        case 'syncing':
          break
        case 'waitDueToError':
        case 'waitDueToThrottle':
          state.dispose()
          toSyncedStateRef.current()
          break
        default:
          noCase(state)
      }
    }

    debugLastChangeRef.current = { ...debugLastChangeRef.current, trace: 'end' }
  }, [state, input, effectTriggerDelay])

  // ---- CLEANUP EFFECT ----
  const handleDispose = () => {
    switch (state.type) {
      case 'synced':
        return
      case 'waitDueToError':
      case 'waitDueToThrottle':
      case 'syncing':
        state.dispose()
        return
      default:
        noCase(state)
    }
  }
  const handleDisposeRef = useRef(handleDispose)
  handleDisposeRef.current = handleDispose
  useEffect(() => {
    handleDisposeRef.current()
  }, [])

  // ---- OUTPUT STATE ----
  // This is a bit tricky because we want to immediately reflect the state
  // change that an input change will cause. This is necessary because rebasing
  // uses this to determine if it should run or not. If we don't update
  // immediately, rebasing and syncing can trigger simultaneously, which in an
  // invariant violation.
  return useMemo((): ServerSyncState => {
    const synced = { type: 'synced' } as const
    switch (state.type) {
      case 'synced':
        if (!input) return synced
        return {
          type: 'syncing',
          startTime: Date.now(),
          failures: [],
          inputSummary: input.changeSummary,
          nextInputSummary: null,
        }
      case 'syncing':
        return {
          type: 'syncing',
          startTime: state.startTime,
          failures: state.failures,
          inputSummary: state.inputSummary,
          nextInputSummary: state.getInputSummaryAfterSync(workingPlan),
        }
      case 'waitDueToError':
        return input === null
          ? synced
          : {
              type: 'waitDueToError',
              waitEndTime: state.waitEndTime,
              failures: state.failures,
              queuedSummary: input.changeSummary,
              retryNow: () => handleClearErrorRef.current(),
            }
      case 'waitDueToThrottle':
        return input === null
          ? synced
          : {
              type: 'waitDueToThrottle',
              waitEndTime: state.waitEndTime,
              queuedSummary: input.changeSummary,
              runNow: () => handleClearThrottleRef.current(),
            }
      default:
        noCase(state)
    }
  }, [input, state, workingPlan])
}

// Use custom memoization because useMemo() is not guaranteed and we need
// the guarantee for correctness.
let _getInputPrev = null as {
  args: any
  result: any
  whatChanged: number[]
} | null
type _Input = Exclude<ReturnType<typeof _getInputMemoized>, null>
const _getInputMemoized = (
  serverPlan: {
    planParamsPostBaseIds: readonly string[]
    reverseHeadIndex: number
  },
  workingPlan: WorkingPlanInfo['workingPlan'],
) => {
  type _Input = typeof result
  assert(_getInputPrev === null || _getInputPrev.args.length === 2)
  const whatChanged = block(() => {
    if (!_getInputPrev) return [0, 1]
    const result: number[] = []
    if (_getInputPrev.args[0] !== serverPlan) result.push(0)
    if (_getInputPrev.args[1] !== workingPlan) result.push(1)
    return result
  })
  if (whatChanged.length === 0) return fGet(_getInputPrev).result as _Input

  const cutAfterIndex = _.findLastIndex(
    serverPlan.planParamsPostBaseIds,
    (x, i) => x === workingPlan.planParamsPostBase[i]?.id,
  )

  assert(cutAfterIndex >= 0)

  const cutAfterId = serverPlan.planParamsPostBaseIds[cutAfterIndex]
  const add = workingPlan.planParamsPostBase.slice(cutAfterIndex + 1).map(
    (x): UserPlanSyncAddInput => ({
      id: x.id,
      params: JSON.stringify(x.params),
      change: JSON.stringify(x.change),
    }),
  )
  const reverseHeadIndex = workingPlan.reverseHeadIndex
  const newParamsHistoryPostBaseIds = [
    ...serverPlan.planParamsPostBaseIds.slice(0, cutAfterIndex + 1),
    ...add.map((x) => x.id),
  ]

  const deleteCount =
    serverPlan.planParamsPostBaseIds.length - 1 - cutAfterIndex
  const result =
    reverseHeadIndex === serverPlan.reverseHeadIndex &&
    deleteCount === 0 &&
    add.length === 0
      ? null
      : {
          cutAfterId,
          add,
          reverseHeadIndex,
          newParamsHistoryPostBaseIds,
          changeSummary: block<
            (
              | { type: 'moveHead' }
              | { type: 'cutBranch'; cutCount: number }
              | { type: 'addItems'; addCount: number }
            )[]
          >(() => {
            if (add.length === 0) {
              // Delete can only happen when we undo and then add, or a merge,
              // which is effectively an undo and add.
              assert(deleteCount === 0)
              return [{ type: 'moveHead' as const }]
            } else {
              return _.compact([
                deleteCount > 0
                  ? { type: 'cutBranch', cutCount: deleteCount }
                  : null,
                { type: 'addItems', addCount: add.length },
                reverseHeadIndex === 0 ? null : { type: 'moveHead' },
              ])
            }
          }),
        }
  _getInputPrev = { args: [serverPlan, workingPlan], result, whatChanged }
  return result
}
