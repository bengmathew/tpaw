import {
  PlanPaths,
  assertFalse,
  fGet,
  getFullDatedDefaultPlanParams,
  getFullDatelessDefaultPlanParams,
  noCase,
} from '@tpaw/common'
import clix from 'clsx'
import React, { useState } from 'react'
import { useMutation } from 'react-relay'
import { graphql } from 'relay-runtime'
import { Spinner } from '../../../../../../Utils/View/Spinner'
import { AppError } from '../../../../../App/AppError'
import {
  useDefaultErrorHandlerForNetworkCall,
  useSetGlobalError,
} from '../../../../../App/GlobalErrorBoundary'
import { User, useUser } from '../../../../../App/WithUser'
import { CenteredModal } from '../../../../../Common/Modal/CenteredModal'
import { useIANATimezoneName } from '../../../../PlanRootHelpers/WithNonPlanParams'
import { PlanMenuActionModalResetMutation } from './__generated__/PlanMenuActionModalResetMutation.graphql'

export const PlanMenuActionModalReset = React.memo(
  ({
    show,
    plan,
    onHide,
    reloadOnSuccess,
    isSyncing,
  }: {
    show: boolean
    plan: User['plans'][number]
    onHide: () => void
    reloadOnSuccess: { planPaths: PlanPaths } | null
    isSyncing: boolean
  }) => {
    return (
      <CenteredModal
        className="w-[600px] dialog-outer-div"
        show={show}
        onOutsideClickOrEscape={null}
      >
        <_Body
          plan={plan}
          onHide={onHide}
          reloadOnSuccess={reloadOnSuccess}
          isSyncing={isSyncing}
        />
      </CenteredModal>
    )
  },
)

const _Body = React.memo(
  ({
    plan,
    onHide,
    reloadOnSuccess,
    isSyncing,
  }: {
    plan: User['plans'][number]
    onHide: () => void
    reloadOnSuccess: { planPaths: PlanPaths } | null
    isSyncing: boolean
  }) => {
    const { defaultErrorHandlerForNetworkCall } =
      useDefaultErrorHandlerForNetworkCall()
    const { setGlobalError } = useSetGlobalError()
    const user = fGet(useUser())
    const { ianaTimezoneName } = useIANATimezoneName()
    // Intentionally not using isRunning from useMutation because we want to
    // the state to stay true even after completion, otherwise it flashes to
    // !isRunning state for a split second before the modal is hidden.
    const [state, setState] = useState<
      { type: 'confirm' } | { type: 'running' } | { type: 'success' }
    >({
      type: 'confirm',
    })

    const [mutation] = useMutation<PlanMenuActionModalResetMutation>(graphql`
      mutation PlanMenuActionModalResetMutation($input: UserPlanResetInput!) {
        userPlanReset(input: $input) {
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

    const handleReset = () => {
      mutation({
        variables: {
          input: {
            userId: user.id,
            planId: plan.id,
            lastSyncAt: plan.lastSyncAt,
            planParams: JSON.stringify(
              plan.isDated
                ? getFullDatedDefaultPlanParams(Date.now(), ianaTimezoneName)
                : getFullDatelessDefaultPlanParams(Date.now()),
            ),
          },
        },
        onCompleted: ({ userPlanReset }) => {
          switch (userPlanReset.__typename) {
            case 'ConcurrentChangeError':
              setGlobalError(new AppError('concurrentChange'))
              return
            case 'PlanAndUserResult':
              const now = Date.now()
              if (userPlanReset.plan.lastSyncAt > now) {
                throw new Error(
                  `lastSyncAt is in the future: ${userPlanReset.plan.lastSyncAt} > ${now}`,
                )
              }
              if (reloadOnSuccess) {
                // View will be removed from under us, so no need to setState.
                const url = reloadOnSuccess.planPaths()
                new URL(window.location.href).searchParams.forEach((v, k) =>
                  url.searchParams.set(k, v),
                )
                window.location.href = url.toString()
              } else {
                setState({ type: 'success' })
              }
              break
            case '%other':
              assertFalse()
            default:
              noCase(userPlanReset)
          }
        },
        onError: (e) => {
          defaultErrorHandlerForNetworkCall({
            e,
            toast: 'Error resetting plan',
          })
          setState({ type: 'confirm' })
        },
      })
    }
    const handleResetRef = React.useRef(handleReset)
    handleResetRef.current = handleReset
    React.useEffect(() => {
      if (!(state.type === 'running' && !isSyncing)) {
        return
      }
      handleResetRef.current()
    }, [isSyncing, state.type])

    return (
      <>
        <h2 className=" dialog-heading">Reset Plan</h2>
        {state.type === 'confirm' || state.type === 'running' ? (
          <>
            <div className=" dialog-content-div relative">
              <p className="p=base">
                Are you sure you want to reset this plan? This cannot be undone.
              </p>
            </div>
            <div className="dialog-button-div">
              <button
                className=" dialog-button-cancel"
                onClick={onHide}
                disabled={state.type === 'running'}
              >
                Cancel
              </button>
              <button
                className="relative dialog-button-warning"
                disabled={state.type === 'running'}
                onClick={() => setState({ type: 'running' })}
              >
                <span className={clix(state.type === 'running' && 'opacity-0')}>
                  Reset
                </span>
                {state.type === 'running' && <Spinner size="text-xl" />}
              </button>
            </div>
          </>
        ) : (
          <>
            <div className=" dialog-content-div relative">
              <p className="p=base">Plan reset successfully.</p>
            </div>

            <div className="dialog-button-div">
              <button className="relative dialog-button-dark" onClick={onHide}>
                Close
              </button>
            </div>
          </>
        )}
      </>
    )
  },
)
