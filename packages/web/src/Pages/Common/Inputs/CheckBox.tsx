import {faCheck} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import {Switch} from '@headlessui/react'
import React, {Dispatch, SetStateAction} from 'react'

export const CheckBox = React.memo(
  ({
    className = '',
    enabled,
    setEnabled,
  }: {
    className?: string
    enabled: boolean
    setEnabled: (enabled:boolean)=>void
  }) => {
    return (
      <Switch
        checked={enabled}
        onChange={setEnabled}
        className={`${className} 
        relative inline-flex items-center justify-center
        ${enabled ? 'bg-gray-700' : 'bg-gray-200'} transition-colors duration-500
        h-[25px] w-[25px] rounded-lg  `}
      >
        <FontAwesomeIcon
          className={`text-gray-100 ${
            enabled ? 'opacity-100' : 'opacity-0'
          } transition-opacity duration-500`}
          icon={faCheck}
        />
      </Switch>
    )
  }
)
