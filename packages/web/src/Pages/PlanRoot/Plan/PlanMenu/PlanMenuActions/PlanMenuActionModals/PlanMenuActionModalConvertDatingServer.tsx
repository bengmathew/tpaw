import React, { useState } from 'react'
import { CenteredModal } from '../../../../../Common/Modal/CenteredModal'
import { PlanMenuActionModalConvertDatingCommon } from './PlanMenuActionModalConvertDatingCommon'
import { PlanParamsHelperFns } from '../../../../../../UseSimulator/PlanParamsHelperFns'
import { useIANATimezoneName } from '../../../../PlanRootHelpers/WithNonPlanParams'
import {
  SimulationInfoForServerSrc,
  useSimulation,
} from '../../../../PlanRootHelpers/WithSimulation'
import { assertFalse, fGet, noCase } from '@tpaw/common'
import { useMutation } from 'react-relay'
import { graphql } from 'relay-runtime'
import { AppError } from '../../../../../App/AppError'
import { PlanMenuActionModalResetMutation } from './__generated__/PlanMenuActionModalResetMutation.graphql'
import {
  useDefaultErrorHandlerForNetworkCall,
  useSetGlobalError,
} from '../../../../../App/GlobalErrorBoundary'
import { useUser } from '../../../../../App/WithUser'

export const PlanMenuActionModalConvertDatingServer = React.memo(
  ({
    show,
    onHide,
    plan,
    isSyncing,
  }: {
    show: boolean
    onHide: () => void
    plan: SimulationInfoForServerSrc['plan']
    isSyncing: boolean
  }) => {
    return (
      <CenteredModal show={show} onOutsideClickOrEscape={null}>
        <_Body plan={plan} onHide={onHide} isSyncing={isSyncing} />
      </CenteredModal>
    )
  },
)

const _Body = React.memo(
  ({
    plan,
    onHide,
    isSyncing,
  }: {
    plan: SimulationInfoForServerSrc['plan']
    onHide: () => void
    isSyncing: boolean
  }) => {
    const user = fGet(useUser())
    const { defaultErrorHandlerForNetworkCall } =
      useDefaultErrorHandlerForNetworkCall()
    const { setGlobalError } = useSetGlobalError()
    const { ianaTimezoneName } = useIANATimezoneName()
    const { planParamsNorm, currentPortfolioBalanceInfo, planPaths } =
      useSimulation()
    const [state, setState] = useState<
      { type: 'confirm' } | { type: 'running' }
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

    const handleConvert = () => {
      mutation({
        variables: {
          input: {
            userId: user.id,
            planId: plan.id,
            lastSyncAt: plan.lastSyncAt,
            planParams: JSON.stringify(
              PlanParamsHelperFns.switchDating(
                planParamsNorm,
                currentPortfolioBalanceInfo,
                ianaTimezoneName,
              ),
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
              // View will be removed from under us, so no need to setState.
              const url = planPaths()
              new URL(window.location.href).searchParams.forEach((v, k) =>
                url.searchParams.set(k, v),
              )
              window.location.href = url.toString()
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
            toast: 'Error converting plan',
          })
          setState({ type: 'confirm' })
        },
      })
    }
    const handleConvertRef = React.useRef(handleConvert)
    handleConvertRef.current = handleConvert
    React.useEffect(() => {
      if (!(state.type === 'running' && !isSyncing)) {
        return
      }
      handleConvertRef.current()
    }, [isSyncing, state.type])

    return (
      <PlanMenuActionModalConvertDatingCommon
        onHide={onHide}
        onConvert={() => setState({ type: 'running' })}
        isRunning={state.type === 'running'}
      />
    )
  },
)
