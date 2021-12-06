import { faAngleDoubleLeft, faCaretLeft, faChevronLeft, faFastBackward, faStepBackward } from '@fortawesome/pro-light-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import React, {useState} from 'react'
import {useTPAW} from '../App/WithTPAW'
import {ConfirmAlert} from '../Common/Modal/ConfirmAlert'

export const Reset = React.memo(({onReset}: {onReset: () => void}) => {
  const [showConfirm, setShowConfirm] = useState(false)

  return (
    <>
      <button className="" onClick={() => setShowConfirm(true)}>

      <FontAwesomeIcon className="mr-1 " icon={faAngleDoubleLeft} />
        Reset to default
      </button>
      {showConfirm && (
        <ConfirmAlert
          title="Confirm Reset"
          confirmText="Reset"
          isWarning
          onCancel={() => setShowConfirm(false)}
          onConfirm={() => {
            setShowConfirm(false)
            onReset()
          }}
        >
          Are you sure you want to reset the inputs?
        </ConfirmAlert>
      )}
    </>
  )
})
