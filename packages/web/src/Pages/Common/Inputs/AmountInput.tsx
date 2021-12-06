import _ from 'lodash'
import React, { useState } from 'react'

export type AmountInputState = ReturnType<typeof useAmountInputState>
export function useAmountInputState(initial: number | null) {
  const [amountStr, setAmountStr] = useState(initial ? `${initial}` : '')
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
    disabled = false
  }: {
    className?: string
    state: AmountInputState
    onAccept?: (value: number) => void
    disabled?:boolean
  }) => {
    return (
      <input
        className={` ${className} bg-gray-200 rounded-lg py-1.5 px-2 `}
        disabled={disabled}
        type="text"
        pattern="[0-9]"
        inputMode="numeric"
        value={amountStr === null ? '$' : _formatCurrency(amountStr)}
        onKeyDown={e => {
          if (e.key === 'Enter') {
            onAccept(amount)
          }
        }}
        onBlur={() => onAccept(amount)}
        onChange={e => setAmountStr(_clean(e.target.value))}
      />
    )
  }
)

const _formatCurrency = (x: string) =>
  `$${_.chunk(x.split('').reverse(), 3)
    .reverse()
    .map(x => x.reverse().join(''))
    .join(',')}`

const _clean = (x: string) => x.trim().replaceAll(/[^\d]/g, '')

const _parse = (cleaned: string) => {
  return cleaned.length === 0 ? 0 : parseInt(cleaned)
}
