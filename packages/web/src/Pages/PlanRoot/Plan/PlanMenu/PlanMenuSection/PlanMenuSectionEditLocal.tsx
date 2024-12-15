import {
  faCalendar,
  faEraser,
  faInfinity,
  faTag,
} from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Menu } from '@headlessui/react'
import {
  PlanParams,
  getFullDatedDefaultPlanParams,
  getFullDatelessDefaultPlanParams,
} from '@tpaw/common'
import { useState } from 'react'
import { CenteredModal } from '../../../../Common/Modal/CenteredModal'
import { useIANATimezoneName } from '../../../PlanRootHelpers/WithNonPlanParams'
import { useSimulationResultInfo } from '../../../PlanRootHelpers/WithSimulation'
import { PlanMenuActionModalConvertDatingLocal } from '../PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalConvertDatingLocal'
import { PlanMenuActionModalLabelInput } from '../PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalLabelInput'
import { PlanMenuActionModalResetLocal } from '../PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalResetLocal'

export const convertToDatelessLabel = 'Convert to Dateless'
export const usePlanMenuSectionEditLocal = ({
  editLabelInfo,
  resetTo,
}: {
  editLabelInfo: {
    curr: string | null
    editLabel: (label: string) => void
  } | null
  resetTo: {
    params: (planParams: PlanParams) => void
    original: { isModified: boolean; reset: () => void } | null
  }
}) => {
  const { ianaTimezoneName } = useIANATimezoneName()

  const { simulationResult } = useSimulationResultInfo()

  const [showEditLabelModal, setShowEditLabelModal] = useState(false)
  const [showConvertDatingModal, setShowConvertDatingModal] = useState(false)
  const [showResetModal, setShowResetModal] = useState(false)
  const [showUndoChangesModal, setShowUndoChangesModal] = useState(false)

  const menuItems = (
    <div className="context-menu-section">
      <h2 className=" context-menu-section-heading">Modify Plan</h2>
      {editLabelInfo && (
        <Menu.Item
          as="button"
          className="context-menu-item"
          onClick={() => setShowEditLabelModal(true)}
        >
          <span className="context-menu-icon">
            <FontAwesomeIcon icon={faTag} />
          </span>{' '}
          Edit Label
        </Menu.Item>
      )}
      {
        <Menu.Item
          as="button"
          className="context-menu-item"
          onClick={() => setShowConvertDatingModal(true)}
        >
          <span className="context-menu-icon ">
            <FontAwesomeIcon
              icon={
                simulationResult.planParamsNormOfResult.datingInfo.isDated
                  ? faInfinity
                  : faCalendar
              }
            />
          </span>{' '}
          {simulationResult.planParamsNormOfResult.datingInfo.isDated
            ? convertToDatelessLabel
            : 'Convert to  Dated'}
        </Menu.Item>
      }
      {resetTo.original ? (
        resetTo.original.isModified && (
          <Menu.Item
            as="button"
            className="context-menu-item text-errorFG"
            onClick={() => setShowUndoChangesModal(true)}
          >
            <span className="context-menu-icon">
              <FontAwesomeIcon icon={faEraser} />
            </span>{' '}
            Undo Changes
          </Menu.Item>
        )
      ) : (
        <Menu.Item
          as="button"
          className="context-menu-item text-errorFG"
          onClick={() => setShowResetModal(true)}
        >
          <span className="context-menu-icon">
            <FontAwesomeIcon icon={faEraser} />
          </span>{' '}
          Reset
        </Menu.Item>
      )}
    </div>
  )

  const modals = (
    <>
      {editLabelInfo && (
        <CenteredModal
          className="dialog-outer-div"
          show={showEditLabelModal}
          onOutsideClickOrEscape={null}
        >
          <PlanMenuActionModalLabelInput
            title="Edit Label"
            initialLabel={editLabelInfo.curr ?? ''}
            buttonLabel={'Edit'}
            onCancel={() => setShowEditLabelModal(false)}
            onAction={(label) => {
              setShowEditLabelModal(false)
              editLabelInfo.editLabel(label)
            }}
            isRunning={false}
          />
        </CenteredModal>
      )}
      <PlanMenuActionModalConvertDatingLocal
        show={showConvertDatingModal}
        onHide={() => setShowConvertDatingModal(false)}
        onConvert={resetTo.params}
        skipNoUndoCopy={!!resetTo.original}
      />
      <PlanMenuActionModalResetLocal
        show={showResetModal}
        onHide={() => setShowResetModal(false)}
        title="Reset"
        message="Are you sure you want to reset this plan? This cannot be undone."
        reset={() =>
          resetTo.params(
            simulationResult.planParamsNormOfResult.datingInfo.isDated
              ? getFullDatedDefaultPlanParams(Date.now(), ianaTimezoneName)
              : getFullDatelessDefaultPlanParams(Date.now()),
          )
        }
      />
      {resetTo.original && (
        <PlanMenuActionModalResetLocal
          show={showUndoChangesModal}
          onHide={() => setShowUndoChangesModal(false)}
          title="Undo Changes"
          message="Are you sure you want to undo the changes you made? This cannot be undone."
          reset={resetTo.original.reset}
        />
      )}
    </>
  )

  return { menuItems, modals }
}
