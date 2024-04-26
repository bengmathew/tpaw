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
import { useURLParam } from '../../../../../../Utils/UseURLParam'
import { useURLUpdater } from '../../../../../../Utils/UseURLUpdater'

export const PlanMenuActionModalConvertDatingServer = React.memo(
  ({
    show,
    onHide,
    plan,
    isSyncing,
    reload,
  }: {
    show: boolean
    onHide: () => void
    plan: SimulationInfoForServerSrc['plan']
    isSyncing: boolean
    reload: () => void
  }) => {
    return (
      <CenteredModal show={show} onOutsideClickOrEscape={null}>
        <_Body
          plan={plan}
          onHide={onHide}
          isSyncing={isSyncing}
          reload={reload}
        />
      </CenteredModal>
    )
  },
)

const _Body = React.memo(
  ({
    plan,
    onHide,
    isSyncing,
    reload,
  }: {
    plan: SimulationInfoForServerSrc['plan']
    onHide: () => void
    isSyncing: boolean
    reload: () => void
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
    const urlUpdater = useURLUpdater()

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
          console.dir(userPlanReset)
          switch (userPlanReset.__typename) {
            case 'ConcurrentChangeError':
              setGlobalError(new AppError('concurrentChange'))
              return
            case 'PlanAndUserResult':
              // View will be removed from under us, so no need to setState.
              const url = planPaths()
              new URL(window.location.href).searchParams.forEach((v, k) =>
                url.searchParams.set(k, v),
              )
              urlUpdater.replace(url)
              console.dir('reload')
              reload()
              // window.location.href = url.toString()

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
        skipNoUndoCopy={false}
      />
    )
  },
)
