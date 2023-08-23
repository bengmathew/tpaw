import { captureException } from '@sentry/nextjs'
import React from 'react'
import { graphql, useMutation } from 'react-relay'
import { errorToast } from '../../../../../../Utils/CustomToasts'
import { useURLUpdater } from '../../../../../../Utils/UseURLUpdater'
import { fGet } from '../../../../../../Utils/Utils'
import { User, useUser } from '../../../../../App/WithUser'
import { CenteredModal } from '../../../../../Common/Modal/CenteredModal'
import { PlanMenuActionModalLabelInput } from './PlanMenuActionModalLabelInput'
import { PlanMenuActionModalEditLabelMutation } from './__generated__/PlanMenuActionModalEditLabelMutation.graphql'

export const PlanMenuActionModalEditLabel = React.memo(
  ({
    plan,
    show,
    onHide,
  }: {
    plan: User['plans'][0]
    show: boolean
    onHide: () => void
  }) => {
    return (
      <CenteredModal
        className="dialog-outer-div"
        show={show}
        onOutsideClickOrEscape={null}
      >
        <_Body plan={plan} onHide={onHide} />
      </CenteredModal>
    )
  },
)

const _Body = React.memo(
  ({ plan, onHide }: { plan: User['plans'][0]; onHide: () => void }) => {
    const user = fGet(useUser())
    const urlUpdater = useURLUpdater()
    const [updateMutation, isUpdateRunning] =
      useMutation<PlanMenuActionModalEditLabelMutation>(graphql`
        mutation PlanMenuActionModalEditLabelMutation(
          $input: UserPlanUpdateInput!
        ) {
          userPlanUpdate(input: $input) {
            slug
            ...PlanWithoutParamsFragment
          }
        }
      `)

    const handleUpdate = (label: string) => {
      updateMutation({
        variables: {
          input: {
            userId: user.id,
            planId: plan.id,
            setLabel: label,
          },
        },
        onCompleted: ({ userPlanUpdate }) => {
          const url = new URL(window.location.href)
          if (url.searchParams.has('plan')) {
            url.searchParams.set('plan', userPlanUpdate.slug)
            urlUpdater.replace(url)
          }
          onHide()
        },
        onError: (e) => {
          captureException(e)
          errorToast('Error updating plan')
        },
      })
    }

    const isRunning = isUpdateRunning
    return (
      <PlanMenuActionModalLabelInput
        title="Edit Label"
        initialLabel={plan.label ?? ''}
        buttonLabel="Save"
        onCancel={onHide}
        onAction={handleUpdate}
        isRunning={isRunning}
      />
    )
  },
)
