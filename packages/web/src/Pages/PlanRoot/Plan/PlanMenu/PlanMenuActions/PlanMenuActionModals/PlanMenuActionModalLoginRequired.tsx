import React, { useEffect, useState } from 'react'
import { appPaths } from '../../../../../../AppPaths'
import { useURLUpdater } from '../../../../../../Utils/UseURLUpdater'
import { CenteredModal } from '../../../../../Common/Modal/CenteredModal'

export const PlanMenuActionModalLoginRequired = React.memo(
  ({
    state,
    onHide,
  }: {
    state: { heading: string; message: string } | null
    onHide: () => void
  }) => {
    const [lastNonNullState, setLastNonNullState] = useState(state)
    useEffect(() => {
      if (!state) return
      setLastNonNullState(state)
    }, [state])
    const urlUpdater = useURLUpdater()
    return (
      <CenteredModal
        className=" dialog-outer-div"
        show={state !== null}
        onOutsideClickOrEscape={null}
      >
        <h2 className=" dialog-heading">{lastNonNullState?.heading ?? ''}</h2>
        <div className=" dialog-content-div">
          <p className="p-base">{lastNonNullState?.message ?? ''}</p>
        </div>
        <div className=" dialog-button-div">
          <button className=" dialog-button-cancel" onClick={onHide}>
            Cancel
          </button>
          <button
            className=" dialog-button-dark"
            onClick={() => urlUpdater.push(appPaths.login(appPaths.plan()))}
          >
            Login / Sign Up
          </button>
        </div>
      </CenteredModal>
    )
  },
)
