import { faCheck } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Switch } from '@headlessui/react'
import clsx from 'clsx'
import React from 'react'

export const SwitchAsCheckBox = React.memo(
  ({
    className = '',
    checked,
    disabled,
    setChecked,
  }: {
    className?: string
    checked: boolean
    disabled?: boolean
    setChecked: (enabled: boolean) => void
  }) => (
    <Switch
      disabled={disabled}
      checked={checked}
      onChange={setChecked}
      className={clsx(
        className,
        'relative inline-flex items-center justify-center',
        'disabled:lighten-2',
        checked ? 'bg-gray-700' : 'bg-gray-200',
        ' transition-colors duration-500',
        'h-[25px] w-[25px] rounded-lg',
      )}
    >
      <FontAwesomeIcon
        className={clsx(
          'text-gray-100  transition-opacity duration-500',
          checked ? 'opacity-100' : 'opacity-0',
        )}
        icon={faCheck}
      />
    </Switch>
  ),
)
