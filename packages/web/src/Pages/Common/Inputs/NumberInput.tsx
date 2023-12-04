import { faMinus, faPlus } from '@fortawesome/pro-regular-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import clix from 'clsx'
import React, { useEffect, useState } from 'react'
import { AmountInput } from './AmountInput'

export const NumberInput = React.memo(
  ({
    value,
    label,
    setValue,
    className = '',
    buttonClassName = 'pl-3 pr-3',
    width = 45,
    modalLabel,
    disabled = false,
    showDecrement = true,
    textAlign = 'center',
    decrement = (x) => x - 1,
    increment = (x) => x + 1,
  }: {
    value: number
    label?: string
    setValue: (value: number) => boolean // true if clamped.
    buttonClassName?: string
    className?: string
    width?: number
    textAlign?: 'left' | 'center' | 'right'
    modalLabel: string | null
    disabled?: boolean
    showDecrement?: boolean
    decrement?: (value: number) => number
    increment?: (value: number) => number
  }) => {
    const [error, setError] = useState(false)
    useEffect(() => {
      const timeout = window.setTimeout(() => setError(false), 1000)
      return () => window.clearTimeout(timeout)
    }, [error])

    const handleAccept = (newValue: number) => {
      setError(setValue(newValue))
    }

    return (
      <div className={`${className} flex items-stretch`}>
        <AmountInput
          className={clix(
            'rounded-lg py-0.5 px-2 mr-3 transition-all duration-1000',
            textAlign === 'left'
              ? 'text-start'
              : textAlign === 'right'
              ? 'text-end'
              : 'text-center',
            error ? 'bg-errorBlockBG' : 'bg-gray-200',
          )}
          style={{ width: `${width}px` }}
          modalLabel={modalLabel}
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
          onClick={() => handleAccept(increment(value))}
          disabled={disabled}
        >
          <FontAwesomeIcon icon={faPlus} />
        </button>
        {showDecrement && (
          <button
            disabled={disabled}
            className={buttonClassName}
            onClick={() => handleAccept(decrement(value))}
          >
            <FontAwesomeIcon icon={faMinus} />
          </button>
        )}
      </div>
    )
  },
)
