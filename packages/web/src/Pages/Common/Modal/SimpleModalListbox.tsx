import React from 'react'
import clsx from 'clsx'
import { Listbox } from '@headlessui/react'
import { ContextModal } from './ContextModal'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faCaretDown, faCheck } from '@fortawesome/pro-solid-svg-icons'

export const SimpleModalListbox = <T,>({
  className,
  value,
  choices,
  onChange,
  getLabel,
}: {
  className?: string
  value: T
  choices: readonly T[]
  onChange: (x: T) => void
  getLabel: (x: T) => string
}) => {
  return (
    <Listbox value={value} onChange={onChange}>
      {({ open }) => (
        <ContextModal align="left" open={open}>
          {({ ref }) => (
            <Listbox.Button className={clsx(className)} ref={ref}>
              {getLabel(value)}
              <FontAwesomeIcon className="ml-2" icon={faCaretDown} />
            </Listbox.Button>
          )}
          <Listbox.Options className="context-menu-outer-div">
            {choices.map((choice, i) => (
              <Listbox.Option
                key={i}
                className="context-menu-item"
                value={choice}
              >
                {({ selected }) => (
                  <div className="flex">
                    <div className="w-[25px]">
                      {selected && <FontAwesomeIcon icon={faCheck} />}
                    </div>
                    {getLabel(choice)}
                  </div>
                )}
              </Listbox.Option>
            ))}
          </Listbox.Options>
        </ContextModal>
      )}
    </Listbox>
  )
}
