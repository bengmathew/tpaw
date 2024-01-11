import React from 'react'
import { CenteredModal } from '../Common/Modal/CenteredModal'

export const PlanRootLinkUnsavedWarningAlert = React.memo(
  ({
    show,
    onCancel,
    onLeave,
  }: {
    show: boolean
    onCancel: () => void
    onLeave: () => void
  }) => {
    return (
      <CenteredModal
        className="dialog-outer-div"
        // If it is a browser nav, the browser alert is good enough.
        show={show}
        onOutsideClickOrEscape={null}
      >
        <h2 className=" dialog-heading">Unsaved Changes</h2>
        <div className=" dialog-content-div relative ">
          <p className=" p-base">
            You have made changes to this plan that you originally loaded from a link. Leaving this page will discard
            these changes.
          </p>
          <p className=" p-base mt-4">
            If you would like to save your changes, select {`"`}
            <span className="font-bold">Save Plan to Account</span>
            {`"`} from the plan menu before leaving the page.
          </p>
        </div>
        <div className=" dialog-button-div">
          <button className=" dialog-button-cancel" onClick={onCancel}>
            Cancel
          </button>
          <button className=" dialog-button-warning" onClick={onLeave}>
            Leave Page
          </button>
        </div>
      </CenteredModal>
    )
  },
)
