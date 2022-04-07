import React from 'react'
import {ValueForYearRange, Year} from '../../TPAWSimulator/TPAWParams'
import {formatCurrency} from '../../Utils/FormatCurrency'
import {trimAndNullify} from '../../Utils/TrimAndNullify'
import {YearRangeDisplay} from './YearRangeDisplay'

export const ValueForYearRangeDisplay = React.memo(
  ({
    entry,
    range,
  }: {
    entry: ValueForYearRange
    range: {start: Year; end: Year} | null
  }) => {
    return (
      <>
        <span>{`${
          trimAndNullify(entry.label) ?? '<no label>'
        }: ${formatCurrency(entry.value)} per year. ${
          entry.nominal ? 'Nominal dollars' : 'Real dollars'
        }. `}</span>
        <YearRangeDisplay value={entry.yearRange} range={range} lengthInBrackets={false} />
      </>
    )
  }
)
