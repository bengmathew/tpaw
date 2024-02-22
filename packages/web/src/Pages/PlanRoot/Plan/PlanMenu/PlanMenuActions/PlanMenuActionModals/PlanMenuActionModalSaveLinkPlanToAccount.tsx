import { fGet } from '@tpaw/common'
import React from 'react'
import { graphql, useMutation } from 'react-relay'
import { appPaths } from '../../../../../../AppPaths'
import { useURLUpdater } from '../../../../../../Utils/UseURLUpdater'
import { useDefaultErrorHandlerForNetworkCall } from '../../../../../App/GlobalErrorBoundary'
import { useUser } from '../../../../../App/WithUser'
import { CenteredModal } from '../../../../../Common/Modal/CenteredModal'
import {
    SimulationInfoForLinkSrc,
    SimulationInfoForPlanMode,
} from '../../../../PlanRootHelpers/WithSimulation'
import { PlanMenuActionModalLabelInput } from './PlanMenuActionModalLabelInput'
import { PlanMenuActionModalSaveLinkPlanToAccountMutation } from './__generated__/PlanMenuActionModalSaveLinkPlanToAccountMutation.graphql'

export const PlanMenuActionModalSaveLinkPlanToAccount = React.memo(
  ({
    show,
    onHide,
    simulationInfoForLinkSrc,
    simulationInfoForPlanMode,
  }: {
    show: boolean
    onHide: () => void
    simulationInfoForPlanMode: SimulationInfoForPlanMode
    simulationInfoForLinkSrc: SimulationInfoForLinkSrc
  }) => {
    return (
      <CenteredModal
        className="dialog-outer-div"
        show={show}
        onOutsideClickOrEscape={null}
      >
        <_Body
          onHide={onHide}
          simulationInfoForLinkSrc={simulationInfoForLinkSrc}
          simulationInfoForPlanMode={simulationInfoForPlanMode}
        />
      </CenteredModal>
    )
  },
)

// Body seperated out to recreate it at each show to reset state.
const _Body = React.memo(
  ({
    onHide,
    simulationInfoForLinkSrc,
    simulationInfoForPlanMode,
  }: {
    onHide: () => void
    simulationInfoForLinkSrc: SimulationInfoForLinkSrc
    simulationInfoForPlanMode: SimulationInfoForPlanMode
  }) => {
    const { defaultErrorHandlerForNetworkCall } =
      useDefaultErrorHandlerForNetworkCall()
    const { setForceNav } = simulationInfoForLinkSrc
    const { planParamsUndoRedoStack } = simulationInfoForPlanMode
    const user = fGet(useUser())

    const urlUpdater = useURLUpdater()

    const [mutation, isRunning] =
      useMutation<PlanMenuActionModalSaveLinkPlanToAccountMutation>(graphql`
        mutation PlanMenuActionModalSaveLinkPlanToAccountMutation(
          $input: UserPlanCreateInput!
        ) {
          userPlanCreate(input: $input) {
            plan {
              slug
              ...PlanWithoutParamsFragment
            }
            user {
              ...WithUser_user
            }
          }
        }
      `)

    const handleCopy = (label: string) => {
      mutation({
        variables: {
          input: {
            userId: user.id,
            label,
            plan: {
              planParamsHistory: [
                ...planParamsUndoRedoStack.undos,
                ...planParamsUndoRedoStack.redos,
              ].map((x) => ({
                id: x.id,
                params: JSON.stringify(x.params),
                change: JSON.stringify(x.change),
              })),
              reverseHeadIndex: planParamsUndoRedoStack.redos.length,
            },
          },
        },
        onCompleted: ({ userPlanCreate }) => {
          setForceNav()
          const url = appPaths['alt-plan']()
          url.searchParams.set('plan', userPlanCreate.plan.slug)
          // Timeout needed for setForceNav() to take effect.
          window.setTimeout(() => urlUpdater.push(url), 1)
        },
        onError: (e) => {
          defaultErrorHandlerForNetworkCall({ e, toast: 'Error copying plan.' })
          onHide()
        },
      })
    }

    return (
      <PlanMenuActionModalLabelInput
        title="Save Plan to Account"
        initialLabel=""
        buttonLabel="Save Plan"
        onCancel={onHide}
        onAction={handleCopy}
        isRunning={isRunning}
      />
    )
  },
)
