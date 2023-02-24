import { faMinus, faPlus } from '@fortawesome/pro-regular-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import React, { useState } from 'react'
import { AmountInput } from './AmountInput'

export const NumberInput = React.memo(
  ({
    value,
    label,
    setValue,
    clamp,
    className = '',
    buttonClassName = 'pl-3 pr-3',
    width = 45,
    modalLabel,
    disabled = false,
  }: {
    value: number
    label?: string
    setValue: (value: number) => void
    clamp: (x: number) => number
    buttonClassName?: string
    className?: string
    width?: number
    modalLabel: string | null
    disabled?: boolean
  }) => {
    const [error, setError] = useState(false)

    const handleAccept = (newValue: number) => {
      const clamped = clamp(newValue)
      if (clamped !== newValue) {
        // Cannot rely on useEffect when external value changes set this, if
        // value is already clamped, it might not actually change externally.
        // and we will be out of sync.
        setError(true)
      }
      setValue(clamped)
    }

    return (
      <div className={`${className} flex items-stretch`}>
        <AmountInput
          className={` rounded-lg py-0.5 px-2 mr-3 text-center transition-all duration-1000
          ${error ? 'bg-errorBlockBG' : 'bg-gray-200'}`}
          style={{ width: `${width}px` }}
          modalLabel={modalLabel}
          onTransitionEnd={() => setError(false)}
          value={value}
          onChange={handleAccept}
          decimals={0}
          disabled={disabled}
        />

        {label && (
          <div className="flex items-center">
            <h2 className="-ml-1">{label}</h2>
          </div>
        )}
        <button
          className={buttonClassName}
          onClick={() => handleAccept(value + 1)}
          disabled={disabled}
        >
          <FontAwesomeIcon icon={faPlus} />
        </button>
        <button
          disabled={disabled}
          className={buttonClassName}
          onClick={() => handleAccept(value - 1)}
        >
          <FontAwesomeIcon icon={faMinus} />
        </button>
      </div>
    )
  },
)
