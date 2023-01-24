import { getDefaultPlanParams } from '@tpaw/common'
import React, { useState } from 'react'
import { useURLUpdater } from '../../../../Utils/UseURLUpdater'
import { useSimulation } from '../../../App/WithSimulation'
import { ConfirmAlert } from '../../../Common/Modal/ConfirmAlert'
import { useGetPlanChartURL } from '../../PlanChart/UseGetPlanChartURL'

export const PlanSummarySaveReset = React.memo(
  ({
    className = '',
    closeMenu,
  }: {
    className?: string
    closeMenu: () => void
  }) => {
    const { setParams } = useSimulation()
    const [showConfirm, setShowConfirm] = useState(false)
    const urlUpdater = useURLUpdater()
    const getPlanChartURL = useGetPlanChartURL()
    return (
      <>
        <button className={`${className}`} onClick={() => setShowConfirm(true)}>
          Clear All Inputs
        </button>
        {showConfirm && (
          <ConfirmAlert
            title="Confirm Reset"
            option1={{
              label: 'Reset',
              onClose: () => {
                setShowConfirm(false)
                setParams(getDefaultPlanParams())
                closeMenu()
                urlUpdater.push(getPlanChartURL('spending-total'))
              },
              isWarning: true,
            }}
            onCancel={() => setShowConfirm(false)}
          >
            Are you sure you want to clear all inputs?
          </ConfirmAlert>
        )}
      </>
    )
  },
)
