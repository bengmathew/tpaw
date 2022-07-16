import _ from 'lodash'
import React, {useEffect, useState} from 'react'
import NumberFormat from 'react-number-format'

export const AmountInput = React.memo(
  ({
    className = '',
    value,
    onChange,
    disabled = false,
    prefix,
    suffix,
    decimals,
  }: {
    className?: string
    value: number
    onChange: (value: number) => void
    disabled?: boolean
    prefix?: string
    suffix?: string
    decimals: number
  }) => {
    const [internalValue, setInternalValue] = useState(value)
    useEffect(() => setInternalValue(value), [value])

    const outputValue = internalValue === null ? 0 : internalValue

    return (
      <NumberFormat
        className={` ${className} `}
        thousandSeparator={true}
        prefix={prefix}
        suffix={suffix}
        value={internalValue}
        decimalScale={decimals}
        fixedDecimalScale
        onValueChange={x => {
          setInternalValue(x.floatValue ?? 0)
        }}
        onBlur={() => onChange(outputValue)}
        onFocus={(e: React.FocusEvent<HTMLInputElement>) =>
          e.target.setSelectionRange(0, e.target.value.length)
        }
        onClick={(e: React.MouseEvent<HTMLInputElement>) =>
          (e.target as HTMLInputElement).setSelectionRange(
            0,
            (e.target as HTMLInputElement).value.length
          )
        }
        onKeyDown={(e: React.KeyboardEvent) => {
          if (e.key === 'Enter') onChange(outputValue)
        }}
      />
    )
  }
)

const _format = (type: 'currency' | 'percent', x: string) =>
  type === 'percent'
    ? x
    : `$${_.chunk(x.split('').reverse(), 3)
        .reverse()
        .map(x => x.reverse().join(''))
        .join(',')}`

const _clean = (x: string) => x.trim().replaceAll(/[^\d]/g, '')

const _parse = (cleaned: string) => {
  return cleaned.length === 0 ? 0 : parseInt(cleaned)
}
