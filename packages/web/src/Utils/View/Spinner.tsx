import { faSpinnerThird } from '@fortawesome/pro-light-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import React from 'react'

export const Spinner = React.memo(
  ({ size = 'text-2xl' }: { size?: 'text-4xl' | 'text-2xl' }) => {
    return (
      <div
        className={`${size} absolute left-[50%] top-[50%] -translate-x-1/2 -translate-y-1/2`}
      >
        <FontAwesomeIcon className="fa-spin " icon={faSpinnerThird} />
      </div>
    )
  },
)
