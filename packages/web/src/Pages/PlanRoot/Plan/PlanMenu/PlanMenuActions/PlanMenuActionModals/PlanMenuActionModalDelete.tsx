import { fGet } from '@tpaw/common'
import clix from 'clsx'
import React, { useState } from 'react'
import { useMutation } from 'react-relay'
import { graphql } from 'relay-runtime'
import { appPaths } from '../../../../../../AppPaths'
import { useURLUpdater } from '../../../../../../Utils/UseURLUpdater'
import { Spinner } from '../../../../../../Utils/View/Spinner'
import { useDefaultErrorHandlerForNetworkCall } from '../../../../../App/GlobalErrorBoundary'
import { User, useUser } from '../../../../../App/WithUser'
import { CenteredModal } from '../../../../../Common/Modal/CenteredModal'
import { PlanMenuActionModalDeleteMutation } from './__generated__/PlanMenuActionModalDeleteMutation.graphql'

export const PlanMenuActionModalDelete = React.memo(
  ({
    show,
    plan,
    onHide,
    toPlansOnSuccess,
    isSyncing,
  }: {
    show: boolean
    plan: User['plans'][number]
    onHide: () => void
    toPlansOnSuccess: boolean
    isSyncing: boolean
  }) => {
    return (
      <CenteredModal
        className=" dialog-outer-div"
        show={show}
        onOutsideClickOrEscape={null}
      >
        <_Body
          plan={plan}
          onHide={onHide}
          toPlansOnSuccess={toPlansOnSuccess}
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
    toPlansOnSuccess,
    isSyncing,
  }: {
    plan: User['plans'][number]
    onHide: () => void
    toPlansOnSuccess: boolean
    isSyncing: boolean
  }) => {
    const { defaultErrorHandlerForNetworkCall } =
      useDefaultErrorHandlerForNetworkCall()
    const user = fGet(useUser())
    const urlUpdater = useURLUpdater()
    // Intentionally not using isRunning from useMutation because we want to
    // the state to stay true even after completion, otherwise it flashes to
    // !isRunning state for a split second before the modal is hidden.
    const [isRunning, setIsRunning] = useState(false)
    const [mutation] = useMutation<PlanMenuActionModalDeleteMutation>(graphql`
      mutation PlanMenuActionModalDeleteMutation($input: UserPlanDeleteInput!) {
        userPlanDelete(input: $input) {
          ...WithUser_user
        }
      }
    `)

    const handleDelete = () => {
      mutation({
        variables: {
          input: {
            userId: user.id,
            planId: plan.id,
          },
        },
        // View will be removed from under us, so no need for setIsRunning(false).
        onCompleted: () => {
          if (toPlansOnSuccess) {
            urlUpdater.replace(appPaths.plans())
          }
        },
        onError: (e) => {
          defaultErrorHandlerForNetworkCall({ e, toast: 'Error deleting plan' })
          setIsRunning(false)
        },
      })
    }
    const handleDeleteRef = React.useRef(handleDelete)
    handleDeleteRef.current = handleDelete
    React.useEffect(() => {
      if (!(isRunning && !isSyncing)) return
      handleDeleteRef.current()
    }, [isRunning, isSyncing])

    return (
      <>
        <h2 className="dialog-heading">Delete Plan</h2>
        <div className=" dialog-content-div">
          <p className="mt-4 p-base">
            {`Are you sure you want to delete this plan? This cannot be undone.`}
          </p>
        </div>
        <div className="dialog-button-div">
          <button
            className=" dialog-button-cancel disabled:lighten"
            onClick={onHide}
            disabled={isRunning}
          >
            Cancel
          </button>
          <button
            className="relative dialog-button-warning"
            disabled={isRunning}
            onClick={() => setIsRunning(true)}
          >
            <span className={clix(isRunning && 'opacity-0')}>Delete</span>
            {isRunning && <Spinner size="text-xl" />}
          </button>
        </div>
      </>
    )
  },
)
