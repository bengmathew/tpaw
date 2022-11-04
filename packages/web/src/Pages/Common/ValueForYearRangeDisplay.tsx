import React from 'react'
import {ValueForYearRange} from '@tpaw/common'
import {formatCurrency} from '../../Utils/FormatCurrency'
import {SimpleRange} from '../../Utils/SimpleRange'
import {trimAndNullify} from '../../Utils/TrimAndNullify'
import {YearRangeDisplay} from './YearRangeDisplay'

export const ValueForYearRangeDisplay = React.memo(
  ({entry, range}: {entry: ValueForYearRange; range: SimpleRange | null}) => {
    return (
      <div>
        <h2>{trimAndNullify(entry.label) ?? '<no label>'}</h2>
        <h2>{`${formatCurrency(entry.value)} per year ${
          entry.nominal ? '(nominal dollars)' : '(real dollars)'
        }`}</h2>
        <YearRangeDisplay
          value={entry.yearRange}
          range={range}
          lengthInBrackets={false}
        />
      </div>
    )
  }
)
