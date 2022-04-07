import {faMinus, faPlus} from '@fortawesome/pro-regular-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import React, {useEffect, useState} from 'react'

export const NumberInput = React.memo(
  ({
    value,
    label,
    setValue,
    clamp,
    className = '',
    buttonClassName = 'pl-3 pr-3',
    showPlusMinus,
    width = 45,
  }: {
    value: number
    label?: string
    setValue: (value: number) => void
    clamp: (x: number) => number
    buttonClassName?: string
    className?: string
    showPlusMinus: boolean
    width?: number
  }) => {
    const [localValue, setLocalValue] = useState<number | null>(value)
    const [error, setError] = useState(false)

    const handleAccept = (newValue: number | null) => {
      if (newValue === null) setLocalValue(value)
      else {
        const clamped = clamp(newValue)
        if (clamped !== newValue) {
          // Cannot rely on useEffect when external value changes set this, if
          // value is already clamped, it might not actually change externally.
          // and we will be out of sync.
          setError(true)
        }
        setLocalValue(clamped)
        setValue(clamped)
      }
    }

    const handleDelta = (delta: number) =>
      handleAccept(localValue !== null ? localValue + delta : 0)

    useEffect(() => {
      setLocalValue(value)
    }, [value])

    return (
      <div className={`${className} flex items-stretch`}>
        <input
          className={` rounded-lg py-0.5 px-2 mr-3 text-center transition-all duration-1000
          ${error ? 'bg-errorBlockBG' : 'bg-gray-200'}`}
          style={{width: `${width}px`}}
          onTransitionEnd={() => setError(false)}
          type="text"
          pattern="[0-9]"
          inputMode="numeric"
          value={localValue === null ? '' : `${localValue}`}
          onKeyDown={e => {
            if (e.key === 'Enter') handleAccept(localValue)
            if (e.key === 'ArrowUp') handleDelta(1)
            if (e.key === 'ArrowDown') handleDelta(-1)
          }}
          onBlur={() => handleAccept(localValue)}
          onChange={e => {
            const strValue = e.target.value.trim()
            if (strValue === '') {
              setLocalValue(null)
            } else {
              const intValue = parseInt(strValue)
              if (isNaN(intValue)) return
              setLocalValue(intValue)
            }
          }}
        />

        {label && (
          <div className="flex items-center">
            <h2 className="-ml-1">{label}</h2>
          </div>
        )}
        {showPlusMinus && (
          <button className={buttonClassName} onClick={() => handleDelta(1)}>
            <FontAwesomeIcon icon={faPlus} />
          </button>
        )}
        {showPlusMinus && (
          <button className={buttonClassName} onClick={() => handleDelta(-1)}>
            <FontAwesomeIcon icon={faMinus} />
          </button>
        )}
      </div>
    )
  }
)
