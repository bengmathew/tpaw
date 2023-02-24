import { ValueForMonthRange } from '@tpaw/common'
import React from 'react'
import { formatCurrency } from '../../Utils/FormatCurrency'
import { SimpleRange } from '../../Utils/SimpleRange'
import { trimAndNullify } from '../../Utils/TrimAndNullify'
import { MonthRangeDisplay } from './MonthRangeDisplay'

export const ValueForMonthRangeDisplay = React.memo(
  ({
    className = '',
    labelClassName = '',
    entry,
    range,
    skipLength,
  }: {
    className?: string
    labelClassName?: string
    entry: ValueForMonthRange
    range: SimpleRange | null
    skipLength: boolean
  }) => {
    return (
      <div className={`${className}`}>
        <h2 className={`${labelClassName}`}>
          {trimAndNullify(entry.label) ?? '<no label>'}
        </h2>
        <h2>{`${formatCurrency(entry.value)} per month ${
          entry.nominal ? '(nominal dollars)' : '(real dollars)'
        }`}</h2>
        <MonthRangeDisplay
          className=""
          value={entry.monthRange}
          range={range}
          skipLength={skipLength}
        />
      </div>
    )
  },
)
