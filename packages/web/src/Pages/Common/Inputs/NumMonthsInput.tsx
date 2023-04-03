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
    rangeAsMFN,
    disabled = false,
  }: {
    className?: string
    modalLabel: string | null
    value: number
    onChange: (value: number) => void
    rangeAsMFN: SimpleRange
    disabled?: boolean
  }) => {
    const numFullYears = numFullYearsIn(value)
    const numRemainingMonths = value % 12
    const fromNumYears = (numYears: number) =>
      numYears * 12 + numRemainingMonths
    const fromNumMonths = (numMonths: number) => numFullYears * 12 + numMonths

    const handleChange = (x: number) => {
      if (_.inRange(x, rangeAsMFN.start, rangeAsMFN.end + 1)) {
        onChange(x)
        return false
      } else {
        onChange(_.clamp(x, rangeAsMFN.start, rangeAsMFN.end))
        return true
      }
    }

    return (
      <div
        className={`${className} grid gap-y-2 gap-x-4`}
        style={{ grid: 'auto/auto 1fr' }}
      >
        <h2 className="">Years</h2>
        <NumberInput
          value={numFullYears}
          setValue={(numYears) => handleChange(fromNumYears(numYears))}
          modalLabel={modalLabel ? `${modalLabel} - Years` : null}
          disabled={disabled}
        />
        <h2 className="">Months</h2>
        <NumberInput
          value={value % 12}
          setValue={(numMonths) => handleChange(fromNumMonths(numMonths))}
          modalLabel={modalLabel ? `${modalLabel} - Months` : null}
          disabled={disabled}
        />
      </div>
    )
  },
)
