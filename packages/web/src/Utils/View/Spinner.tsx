import { faSpinnerThird } from '@fortawesome/pro-light-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import clsx from 'clsx'
import React from 'react'

export const Spinner = React.memo(
  ({
    size = 'text-2xl',
    dontCenterHorizontally = false,
  }: {
    size?: 'text-4xl' | 'text-3xl' | 'text-2xl' | 'text-xl'
    dontCenterHorizontally?: boolean
  }) => {
    return (
      <div
        className={clsx(
          `${size}`,
          'absolute  top-[50%] -translate-y-1/2',
          !dontCenterHorizontally && 'left-[50%] -translate-x-1/2',
        )}
      >
        <FontAwesomeIcon className="fa-spin " icon={faSpinnerThird} />
      </div>
    )
  },
)
