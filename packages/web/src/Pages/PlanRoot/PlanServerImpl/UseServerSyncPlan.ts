import * as Sentry from '@sentry/nextjs'
import { assert, assertFalse, noCase } from '@tpaw/common'
import _ from 'lodash'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useMutation } from 'react-relay'
import { graphql } from 'relay-runtime'
import { errorToast } from '../../../Utils/CustomToasts'
import { AppError } from '../../App/AppError'
import { WorkingPlanInfo } from '../PlanRootHelpers/UseWorkingPlan'
import {
  UseServerSyncPlanMutation,
  UserPlanSyncAddInput,
} from './__generated__/UseServerSyncPlanMutation.graphql'

type _ServerPlan = {
  planId: string
  lastSyncAt: number
  planParamsPostBaseIds: readonly string[]
  reverseHeadIndex: number
}

export const useServerSyncPlan = (
  userId: string,
  serverPlan: _ServerPlan,
  workingPlan: WorkingPlanInfo['workingPlan'],
  setServerPlan: (x: _ServerPlan) => void,
) => {
  const [throttle, setThrottle] = useState(false)
  const [inputKey, setInputKey] = useState(0)
  useEffect(() => {
    if (!throttle) return
    const timeout = window.setTimeout(() => setThrottle(false), 3000)
    return () => window.clearTimeout(timeout)
  }, [throttle])

  const liveInput = useMemo(() => {
    inputKey // Dummy  use to make sure we include it in deps.
    return _getInput(serverPlan, workingPlan)
  }, [serverPlan, workingPlan, inputKey])

  const [commitSync, isRunning] = useMutation<UseServerSyncPlanMutation>(
    graphql`
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
    `,
  )

  const handleSyncEvent = (input: _Input) => {
    const { cutAfterId, add, reverseHeadIndex, newParamsHistoryPostBaseIds } =
      input
    setThrottle(true)
    return commitSync({
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
      onCompleted: ({ userPlanSync }) => {
        switch (userPlanSync.__typename) {
          case 'ConcurrentChangeError':
            throw new AppError('concurrentChange')
          case 'PlanAndUserResult':
            const newServerPlan = {
              planId: serverPlan.planId,
              lastSyncAt: userPlanSync.plan.lastSyncAt,
              planParamsPostBaseIds: newParamsHistoryPostBaseIds,
              reverseHeadIndex,
            }
            setServerPlan(newServerPlan)
            break
          case '%other':
            assertFalse()
          default:
            noCase(userPlanSync)
        }
      },
      onError: (e) => {
        const timeout = 30000
        // Force regeneration of input which will in turn trigger a retry.

        window.setTimeout(() => setInputKey((prev) => prev + 1), timeout)
        Sentry.captureException(e)
        errorToast(
          `Could not save changes to server. Retrying in ${
            timeout / 1000
          } seconds.`,
          {
            toastId: 'serverSyncError',
            autoClose: timeout - 1000, // Let the toast close a bit early.
          },
        )
      },
    })
  }

  const handleSyncEventRef = useRef(handleSyncEvent)
  handleSyncEventRef.current = handleSyncEvent

  const [inputForCommit, setInputForCommit] = useState<_Input | null>(null)
  useEffect(() => {
    if (throttle || isRunning) return
    setInputForCommit(liveInput)
  }, [isRunning, throttle, liveInput])

  //FEATURE: Use a stateful object in a ref for whole sync logic instead of
  // this hack?
  useEffect(() => {
    // Cannot check for throttle or isRunning here because turn true as soon as
    // the event start which will in turn trigger dispose. inputForCommit
    // indirection from liveInput is to get around this.
    if (!inputForCommit) return
    const { dispose } = handleSyncEventRef.current(inputForCommit)
    return () => dispose()
  }, [inputForCommit])
  return liveInput !== null
}

type _Input = Exclude<ReturnType<typeof _getInput>, null>
const _getInput = (
  serverPlan: _ServerPlan,
  workingPlan: WorkingPlanInfo['workingPlan'],
) => {
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

  return reverseHeadIndex === serverPlan.reverseHeadIndex &&
    cutAfterIndex === serverPlan.planParamsPostBaseIds.length - 1 &&
    add.length === 0
    ? null
    : { cutAfterId, add, reverseHeadIndex, newParamsHistoryPostBaseIds }
}
