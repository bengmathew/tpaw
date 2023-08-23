import { Switch } from '@headlessui/react'
import React from 'react'

export const ToggleSwitch = React.memo(
  ({
    className = '',
    checked,
    setChecked,
    disabled,
    type = 'oneSided',
  }: {
    className?: string
    checked: boolean
    setChecked: (value: boolean) => void
    disabled?: boolean
    type?: 'twoSided' | 'oneSided'
  }) => {
    return (
      <Switch
        checked={checked}
        onChange={setChecked}
        disabled={disabled}
        className={`${className} ${
          checked || type === 'twoSided' ? 'bg-darkGray' : 'bg-gray-200'
        } relative inline-flex items-center h-[20px] rounded-full w-[36px] transition-colors `}
      >
        <span
          className={`${
            checked ? 'translate-x-[19px]' : 'translate-x-[3px]'
          } inline-block w-[14px] h-[14px] transform bg-white rounded-full transition-transform`}
        />
      </Switch>
    )
  },
)
