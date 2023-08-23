import React from 'react'
import { useURLUpdater } from '../../../../../../Utils/UseURLUpdater'
import { CenteredModal } from '../../../../../Common/Modal/CenteredModal'
import { useGetSectionURL } from '../../../Plan'
import { setPlanResultsChartURLOn } from '../../../PlanResults/UseGetPlanResultsChartURL'

export const PlanMenuActionModalResetLocal = React.memo(
  ({
    show,
    onHide,
    reset,
    message,
    title,
  }: {
    show: boolean
    onHide: () => void
    reset: () => void
    title: string
    message: string
  }) => {
    const getSectionURL = useGetSectionURL()
    const urlUpdater = useURLUpdater()

    const handleClick = () => {
      urlUpdater.push(
        setPlanResultsChartURLOn(getSectionURL('summary'), 'spending-total'),
      )
      reset()
    }
    return (
      <CenteredModal
        className="dialog-outer-div"
        show={show}
        onOutsideClickOrEscape={null}
      >
        <div className=" dialog-outer-div">
          <h2 className=" dialog-heading">{title}</h2>
          <div className=" dialog-content-div">
            <p className="p-base">{message}</p>
          </div>
          <div className=" dialog-button-div">
            <button className=" dialog-button-cancel" onClick={onHide}>
              Cancel
            </button>
            <button className=" dialog-button-warning" onClick={handleClick}>
              {title}
            </button>
          </div>
        </div>
      </CenteredModal>
    )
  },
)
