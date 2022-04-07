import {Switch} from '@headlessui/react'
import React, {Dispatch, SetStateAction} from 'react'

export const ToggleSwitch = React.memo(
  ({
    className = '',
    enabled,
    setEnabled,
    type = 'oneSided',
  }: {
    className?:string
    enabled: boolean
    setEnabled: (value:boolean)=>void
    type?: 'twoSided' | 'oneSided'
  }) => {
    return (
      <Switch
        checked={enabled}
        onChange={setEnabled}
        className={`${className} ${
          enabled || type === 'twoSided' ? 'bg-darkGray' : 'bg-gray-200'
        } relative inline-flex items-center h-[20px] rounded-full w-[36px] transition-colors `}
      >
        <span
          className={`${
            enabled ? 'translate-x-[19px]' : 'translate-x-[3px]'
          } inline-block w-[14px] h-[14px] transform bg-white rounded-full transition-transform`}
        />
      </Switch>
    )
  }
)
