import { fGet } from '@tpaw/common'
import clix from 'clsx'
import React from 'react'
import { useMutation } from 'react-relay'
import { graphql } from 'relay-runtime'
import { appPaths } from '../../../../../../AppPaths'
import { useURLUpdater } from '../../../../../../Utils/UseURLUpdater'
import { Spinner } from '../../../../../../Utils/View/Spinner'
import { useDefaultErrorHandlerForNetworkCall } from '../../../../../App/GlobalErrorBoundary'
import { User, useUser } from '../../../../../App/WithUser'
import { CenteredModal } from '../../../../../Common/Modal/CenteredModal'
import { PlanMenuActionModalSetAsMainPlanMutation } from './__generated__/PlanMenuActionModalSetAsMainPlanMutation.graphql'

export const PlanMenuActionModalSetAsMain = React.memo(
  ({
    show,
    plan,
    onHide,
    switchToMainPlanOnSuccess,
    isSyncing,
  }: {
    show: boolean
    plan: User['plans'][number]
    onHide: () => void
    switchToMainPlanOnSuccess: boolean
    isSyncing: boolean
  }) => {
    const { defaultErrorHandlerForNetworkCall } =
      useDefaultErrorHandlerForNetworkCall()
    const user = fGet(useUser())
    const urlUpdater = useURLUpdater()
    const [mutation, isRunning] =
      useMutation<PlanMenuActionModalSetAsMainPlanMutation>(graphql`
        mutation PlanMenuActionModalSetAsMainPlanMutation(
          $input: UserPlanSetAsMainInput!
        ) {
          userPlanSetAsMain(input: $input) {
            ...WithUser_user
          }
        }
      `)

    const handleClick = () => {
      mutation({
        variables: {
          input: {
            userId: user.id,
            planId: plan.id,
          },
        },
        onCompleted: () => {
          if (switchToMainPlanOnSuccess) {
            urlUpdater.replace(appPaths.plan())
            onHide()
          } else {
            // View will be removed from under us.
          }
        },
        onError: (e) => {
          defaultErrorHandlerForNetworkCall({ e, toast: 'Error updating plan' })
        },
      })
    }

    return (
      <CenteredModal
        className=" dialog-outer-div"
        show={show}
        onOutsideClickOrEscape={null}
      >
        <h2 className=" dialog-heading">
          {isSyncing ? 'Syncing Plan to Server...' : 'Set as Main Plan'}
        </h2>
        <div className=" dialog-content-div relative">
          <p className={clix('p-base', isSyncing && 'opacity-0')}>
            {`Your current main plan will be moved to "Other Plans."`}
          </p>
          {isSyncing && <Spinner size="text-3xl" />}
        </div>
        <div className=" dialog-button-div">
          <button
            className=" dialog-button-cancel "
            onClick={onHide}
            disabled={isRunning}
          >
            Cancel
          </button>
          <button
            className=" dialog-button-dark relative"
            disabled={isRunning || isSyncing}
            onClick={handleClick}
          >
            <span className={clix(isRunning && 'opacity-0')}>Continue</span>
            {isRunning && <Spinner size="text-xl" />}
          </button>
        </div>
      </CenteredModal>
    )
  },
)
