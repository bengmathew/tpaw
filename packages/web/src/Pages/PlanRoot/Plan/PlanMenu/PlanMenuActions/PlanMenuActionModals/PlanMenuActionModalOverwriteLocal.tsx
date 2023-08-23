import React from 'react'
import * as uuid from 'uuid'
import { appPaths } from '../../../../../../AppPaths'
import { successToast } from '../../../../../../Utils/CustomToasts'
import { useURLUpdater } from '../../../../../../Utils/UseURLUpdater'
import { CenteredModal } from '../../../../../Common/Modal/CenteredModal'
import { WorkingPlanInfo } from '../../../../PlanRootHelpers/UseWorkingPlan'
import { PlanLocalStorage } from '../../../../PlanRootLocalMain/PlanLocalStorage'

export const PlanMenuActionModalOverwriteLocal = React.memo(
  ({
    show,
    onHide,
    planPramsUndoRedoStack,
    setForceNav,
  }: {
    show: boolean
    onHide: () => void
    planPramsUndoRedoStack: WorkingPlanInfo['planParamsUndoRedoStack']
    setForceNav:()=>void
  }) => {
    const urlUpdater = useURLUpdater()
    const handleClick = () => {
      PlanLocalStorage.clear()
      PlanLocalStorage.write({
        v: 1,
        planId: uuid.v4(),
        planParamsPostBase: [
          ...planPramsUndoRedoStack.undos,
          ...planPramsUndoRedoStack.redos,
        ],
        reverseHeadIndex: planPramsUndoRedoStack.redos.length,
      })
      setForceNav()
      urlUpdater.push(appPaths['guest']())
      successToast('Successfully overwrote main plan.')
    }
    return (
      <CenteredModal
        className="w-[500px] dialog-outer-div"
        show={show}
        onOutsideClickOrEscape={null}
      >
        <div className=" dialog-outer-div">
          <h2 className=" dialog-heading">Make This the Main Plan</h2>
          <div className=" dialog-content-div">
            <p className="p-base">
              This will overwrite your main plan with the current plan.
            </p>
          </div>
          <div className=" dialog-button-div">
            <button className=" dialog-button-cancel" onClick={onHide}>
              Cancel
            </button>
            <button className=" dialog-button-warning" onClick={handleClick}>
              Overwrite
            </button>
          </div>
        </div>
      </CenteredModal>
    )
  },
)
