import _ from 'lodash'
import React from 'react'
import { numFullYearsIn } from '../../../Utils/NumFullYearsIn'
import { SimpleRange } from '../../../Utils/SimpleRange'
import { NumberInput } from './NumberInput'

export const NumMonthsInput = React.memo(
  ({
    className = '',
    modalLabel,
    value,
    onChange,
    range,
    disabled = false,
  }: {
    className?: string
    modalLabel: string | null
    value: number
    onChange: (value: number) => void
    range: SimpleRange
    disabled?: boolean
  }) => {
    const numFullYears = numFullYearsIn(value)
    const numRemainingMonths = value % 12
    const fromNumYears = (numYears: number) =>
      numYears * 12 + numRemainingMonths
    const fromNumMonths = (numMonths: number) => numFullYears * 12 + numMonths

    const monthRange = {
      start: range.start - numFullYears * 12,
      end: range.end - numFullYears * 12,
    }

    const yearRange = {
      start: Math.ceil((range.start - numRemainingMonths) / 12),
      end: numFullYearsIn(range.end - numRemainingMonths),
    }

    return (
      <div
        className={`${className} grid gap-y-2 gap-x-4`}
        style={{ grid: 'auto/auto 1fr' }}
      >
        <h2 className="">Years</h2>
        <NumberInput
          value={numFullYears}
          setValue={(numYears) => onChange(fromNumYears(numYears))}
          clamp={(x) => _.clamp(x, yearRange.start, yearRange.end)}
          modalLabel={modalLabel ? `${modalLabel} - Years` : null}
          disabled={disabled}
        />
        <h2 className="">Months</h2>
        <NumberInput
          value={value % 12}
          setValue={(numMonths) => onChange(fromNumMonths(numMonths))}
          clamp={(x) => _.clamp(x, monthRange.start, monthRange.end)}
          modalLabel={modalLabel ? `${modalLabel} - Months` : null}
          disabled={disabled}
        />
      </div>
    )
  },
)
