import React from 'react'
import { SimpleRange } from'@tpaw/common'
import { NumberInput } from '../NumberInput'
import { InMonthsFns } from '@tpaw/common'
import clsx from 'clsx'

export const InMonthsInput = React.memo(
  ({
    className = '',
    modalLabel,
    normValue: { baseValue, validRangeInMonths },
    onChange,
    disabled = false,
  }: {
    className?: string
    modalLabel: string | null
    normValue: {
      baseValue: { inMonths: number }
      // Note this is not in MFN, but in number of months.
      validRangeInMonths: {
        includingLocalConstraints: SimpleRange | null
      }
    }
    onChange: (x: { inMonths: number }) => void
    disabled?: boolean
  }) => {
    const numFullYears = InMonthsFns.getFullYears(baseValue)
    const numRemainingMonths = baseValue.inMonths % 12
    const fromNumYears = (numYears: number) =>
      numYears * 12 + numRemainingMonths
    const fromNumMonths = (numMonths: number) => numFullYears * 12 + numMonths

    const handleChange = (inMonths: number) => {
      if (!validRangeInMonths.includingLocalConstraints) {
        return true
      }
      if (
        SimpleRange.Closed.isIn(
          inMonths,
          validRangeInMonths.includingLocalConstraints,
        )
      ) {
        onChange({ inMonths })
        return false
      } else {
        onChange({
          inMonths: SimpleRange.Closed.clamp(
            inMonths,
            validRangeInMonths.includingLocalConstraints,
          ),
        })
        return true
      }
    }

    return (
      <div
        className={clsx(className, 'grid gap-y-2 gap-x-4')}
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
          value={numRemainingMonths}
          setValue={(numMonths) => handleChange(fromNumMonths(numMonths))}
          modalLabel={modalLabel ? `${modalLabel} - Months` : null}
          disabled={disabled}
        />
      </div>
    )
  },
)
