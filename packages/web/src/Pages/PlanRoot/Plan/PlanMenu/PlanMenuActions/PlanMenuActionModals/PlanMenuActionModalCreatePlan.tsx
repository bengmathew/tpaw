import { captureException } from '@sentry/nextjs'
import {
  PlanParamsChangeActionCurrent,
  fGet,
  getDefaultPlanParams,
} from '@tpaw/common'
import React, { useState } from 'react'
import { useMutation } from 'react-relay'
import { graphql } from 'relay-runtime'
import * as uuid from 'uuid'
import { appPaths } from '../../../../../../AppPaths'
import { errorToast } from '../../../../../../Utils/CustomToasts'
import { useURLUpdater } from '../../../../../../Utils/UseURLUpdater'
import { useUser } from '../../../../../App/WithUser'
import { CenteredModal } from '../../../../../Common/Modal/CenteredModal'
import { useIANATimezoneName } from '../../../../PlanRootHelpers/WithNonPlanParams'
import { PlanMenuActionModalLabelInput } from './PlanMenuActionModalLabelInput'
import { PlanMenuActionModalCreatePlanMutation } from './__generated__/PlanMenuActionModalCreatePlanMutation.graphql'

export const PlanMenuActionModalCreatePlan = React.memo(
  ({
    show,
    onHide,
    switchOnCreate,
  }: {
    show: boolean
    onHide: () => void
    switchOnCreate: { onBeforeNav: (() => void) | null } | false
  }) => {
    const urlUpdater = useURLUpdater()
    return (
      <CenteredModal
        className="w-[400px] dialog-outer-div"
        show={show}
        onOutsideClickOrEscape={null}
      >
        <_Body
          onCancel={onHide}
          onCreate={(slug) => {
            if (switchOnCreate) {
              switchOnCreate.onBeforeNav?.()
              const url = appPaths['alt-plan']()
              url.searchParams.set('plan', slug)
              // Delay to let setForceNav to true take effect.
              window.setTimeout(() => urlUpdater.push(url), 1)
            } else {
              onHide()
            }
          }}
        />
      </CenteredModal>
    )
  },
)

const _Body = React.memo(
  ({
    onCancel,
    onCreate,
  }: {
    onCancel: () => void
    onCreate: (slug: string) => void
  }) => {
    const { ianaTimezoneName } = useIANATimezoneName()
    const user = fGet(useUser())
    // Intentionally not using isRunning from useMutation because we want to
    // the state to stay true even after completion, otherwise it flashes to
    // !isRunning state for a split second before the modal is hidden.
    const [isRunning, setIsRunning] = useState(false)
    const [commit] = useMutation<PlanMenuActionModalCreatePlanMutation>(graphql`
      mutation PlanMenuActionModalCreatePlanMutation(
        $input: UserPlanCreateInput!
      ) {
        userPlanCreate(input: $input) {
          plan {
            id
            label
            slug
            ...PlanWithoutParamsFragment
          }
          user {
            ...WithUser_user
          }
        }
      }
    `)

    const handleCreate = (label: string) => {
      const currentTimestamp = Date.now()
      const change: PlanParamsChangeActionCurrent = {
        type: 'start',
        value: null,
      }
      setIsRunning(true)
      commit({
        variables: {
          input: {
            userId: user.id,
            label,
            plan: {
              planParamsHistory: [
                {
                  id: uuid.v4(),
                  params: JSON.stringify(
                    getDefaultPlanParams(currentTimestamp, ianaTimezoneName),
                  ),
                  change: JSON.stringify(change),
                },
              ],
              reverseHeadIndex: 0,
            },
          },
        },

        // Don't set isRunning to false, let it stay true until the modal is removed.
        onCompleted: ({ userPlanCreate }) => onCreate(userPlanCreate.plan.slug),
        onError: (e) => {
          captureException(e)
          errorToast('Error creating plan')
          setIsRunning(false)
        },
      })
    }
    return (
      <PlanMenuActionModalLabelInput
        title="Create a New Plan"
        initialLabel=""
        buttonLabel="Create"
        onCancel={onCancel}
        onAction={handleCreate}
        isRunning={isRunning}
      />
    )
  },
)
