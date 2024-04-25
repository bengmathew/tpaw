import _ from 'lodash'
import React from 'react'
import { useSimulation } from '../../../../PlanRootHelpers/WithSimulation'
import clsx from 'clsx'
import { Spinner } from '../../../../../../Utils/View/Spinner'

export const PlanMenuActionModalConvertDatingCommon = React.memo(
  ({
    onHide,
    onConvert,
    isRunning,
  }: {
    onHide: () => void
    onConvert: () => void
    isRunning: boolean
  }) => {
    const { planParamsNorm } = useSimulation()
    const { datingInfo } = planParamsNorm
    const targetDatingStr = datingInfo.isDated ? 'dateless' : 'dated'
    return (
      <div className=" dialog-outer-div">
        <h2 className=" dialog-heading">
          Convert to {_.capitalize(targetDatingStr)} Plan
        </h2>
        <div className=" dialog-content-div">
          {datingInfo.isDated && (
            <p className="p-base">
              Dateless plans are not tied to the current date and do not change
              over time. Recommended for examples and not for personal planning.{' '}
            </p>
          )}
          <p className="p-base mt-2">
            Converting this plan to a {targetDatingStr} plan will delete history
            for this plan. This cannot be undone.
          </p>
          <p className="p-base mt-2">Are you sure you want to continue?</p>
        </div>
        <div className=" dialog-button-div">
          <button className=" dialog-button-cancel" onClick={onHide}>
            Cancel
          </button>
          <button
            className=" dialog-button-dark relative"
            disabled={isRunning}
            onClick={onConvert}
          >
            <span className={clsx(isRunning && 'opacity-0')}>Convert</span>
            {isRunning && <Spinner size="text-xl" />}
          </button>
        </div>
      </div>
    )
  },
)
