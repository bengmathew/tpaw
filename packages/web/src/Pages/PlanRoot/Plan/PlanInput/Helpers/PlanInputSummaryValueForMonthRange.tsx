import { ValueForMonthRange, ValueForMonthRanges } from '@tpaw/common'
import React from 'react'
import { SimpleRange } from '../../../../../Utils/SimpleRange'
import { ValueForMonthRangeDisplay } from '../../../../Common/ValueForMonthRangeDisplay'
import _ from 'lodash'

export const PlanInputSummaryValueForMonthRange = React.memo(
  ({
    entries,
    range,
  }: {
    entries: ValueForMonthRanges
    range: SimpleRange
  }) => {
    const asArray = _.values(entries).sort((a, b) => a.sortIndex - b.sortIndex)
    if (asArray.length === 0) return <h2>None</h2>
    return (
      <ol className={`list-outside list-disc ml-3 grid gap-y-2 mt-1`}>
        {asArray.map((x, i) => (
          <li key={i} className="">
            <ValueForMonthRangeDisplay
              entry={x}
              rangeAsMFN={range}
              skipLength={false}
            />
          </li>
        ))}
      </ol>
    )
  },
)
