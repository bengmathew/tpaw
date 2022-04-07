import { faAngleDoubleLeft } from '@fortawesome/pro-light-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import React, { useState } from 'react'
import { getDefaultParams } from '../../../TPAWSimulator/DefaultParams'
import { useSimulation } from '../../App/WithSimulation'
import { ConfirmAlert } from '../../Common/Modal/ConfirmAlert'

export const Reset = React.memo(() => {
  const {setParams} = useSimulation()
  const [showConfirm, setShowConfirm] = useState(false)

  return (
    <>
      <button className="" onClick={() => setShowConfirm(true)}>
        <FontAwesomeIcon className="mr-1 " icon={faAngleDoubleLeft} />
        Reset
      </button>
      {showConfirm && (
        <ConfirmAlert
          title="Confirm Reset"
          confirmText="Reset"
          isWarningButton
          onCancel={() => setShowConfirm(false)}
          onConfirm={() => {
            setShowConfirm(false)
            setParams(getDefaultParams())
          }}
        >
          Are you sure you want to reset the inputs?
        </ConfirmAlert>
      )}
    </>
  )
})
