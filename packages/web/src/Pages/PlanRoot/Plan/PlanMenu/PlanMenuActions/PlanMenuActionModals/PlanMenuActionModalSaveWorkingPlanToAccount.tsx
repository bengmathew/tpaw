import { fGet } from '@tpaw/common'
import React from 'react'
import { graphql, useMutation } from 'react-relay'
import { appPaths } from '../../../../../../AppPaths'
import { useURLUpdater } from '../../../../../../Utils/UseURLUpdater'
import { useDefaultErrorHandlerForNetworkCall } from '../../../../../App/GlobalErrorBoundary'
import { useUser } from '../../../../../App/WithUser'
import { CenteredModal } from '../../../../../Common/Modal/CenteredModal'
import { SimulationInfoForPlanMode } from '../../../../PlanRootHelpers/WithSimulation'
import { PlanMenuActionModalLabelInput } from './PlanMenuActionModalLabelInput'
import { PlanMenuActionModalSaveWorkingPlanToAccountMutation } from './__generated__/PlanMenuActionModalSaveWorkingPlanToAccountMutation.graphql'

export const PlanMenuActionModalSaveWorkingPlanToAccount = React.memo(
  ({
    show,
    onHide,
    setForceNav,
    simulationInfoForPlanMode,
    label,
  }: {
    show: boolean
    onHide: () => void
    simulationInfoForPlanMode: SimulationInfoForPlanMode
    setForceNav: () => void
    label: string | null
  }) => {
    return (
      <CenteredModal
        className="dialog-outer-div"
        show={show}
        onOutsideClickOrEscape={null}
      >
        <_Body
          onHide={onHide}
          setForceNav={setForceNav}
          label={label}
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
    setForceNav,
    simulationInfoForPlanMode,
    label,
  }: {
    onHide: () => void
    setForceNav: () => void
    simulationInfoForPlanMode: SimulationInfoForPlanMode
    label: string | null
  }) => {
    const { defaultErrorHandlerForNetworkCall } =
      useDefaultErrorHandlerForNetworkCall()
    const { planParamsUndoRedoStack } = simulationInfoForPlanMode
    const user = fGet(useUser())

    const urlUpdater = useURLUpdater()

    const [mutation, isRunning] =
      useMutation<PlanMenuActionModalSaveWorkingPlanToAccountMutation>(graphql`
        mutation PlanMenuActionModalSaveWorkingPlanToAccountMutation(
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
                params: JSON.stringify(x.paramsUnmigrated ?? x.params),
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
        initialLabel={label ?? ''}
        buttonLabel="Save Plan"
        onCancel={onHide}
        onAction={handleCopy}
        isRunning={isRunning}
      />
    )
  },
)
