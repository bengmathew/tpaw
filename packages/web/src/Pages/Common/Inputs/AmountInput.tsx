import _ from 'lodash'
import React, {useEffect, useState} from 'react'

export type AmountInputState = ReturnType<typeof useAmountInputState>
export function useAmountInputState(value: number | null) {
  const [amountStr, setAmountStr] = useState(value !== null ? `${value}` : '')
  useEffect(() => setAmountStr(value !== null ? `${value}` : ''), [value])
  const amount = _parse(amountStr)
  return {
    amount,
    amountStr,
    setAmountStr,
  }
}

export const AmountInput = React.memo(
  ({
    className = '',
    state: {amount, amountStr, setAmountStr},
    onAccept = () => {},
    disabled = false,
    type,
  }: {
    className?: string
    state: AmountInputState
    onAccept?: (value: number) => void
    disabled?: boolean
    type: 'currency' | 'percent'
  }) => {
    const symbol = type === 'currency' ? '$' : '%'
    return (
      <input
        className={` ${className} bg-gray-200 rounded-lg py-1.5 px-2 `}
        disabled={disabled}
        type="text"
        pattern="[0-9]"
        inputMode="numeric"
        value={amountStr === null ? symbol : _format(type, amountStr)}
        onKeyDown={e => {
          if (e.key === 'Enter') onAccept(amount)
        }}
        onBlur={() => onAccept(amount)}
        onChange={e => setAmountStr(_clean(e.target.value))}
      />
    )
  }
)

const _format = (type: 'currency' | 'percent', x: string) =>
  `${type === 'currency' ? '$' : ''}${_.chunk(x.split('').reverse(), 3)
    .reverse()
    .map(x => x.reverse().join(''))
    .join(',')}`

const _clean = (x: string) => x.trim().replaceAll(/[^\d]/g, '')

const _parse = (cleaned: string) => {
  return cleaned.length === 0 ? 0 : parseInt(cleaned)
}
